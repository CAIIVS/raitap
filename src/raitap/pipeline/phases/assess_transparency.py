"""Transparency phase — instantiates explainers + collects results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from raitap import raitap_log
from raitap.configs import cfg_to_dict
from raitap.metrics import metrics_prediction_pair
from raitap.transparency.factory import Explanation
from raitap.types import TaskKind

if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig
    from raitap.data import Data
    from raitap.data.preprocessing import ResolvedPreprocessing
    from raitap.models import Model
    from raitap.pipeline.outputs import ForwardOutput
    from raitap.transparency.contracts import InputSpec
    from raitap.transparency.results import ExplanationResult


def resolve_explainer_runtime_kwargs(
    explainer_config: Any,
    *,
    forward_output: torch.Tensor,
) -> dict[str, Any]:
    """Resolve runtime kwargs for an explainer config.

    Currently handles the ``target: auto_pred`` sentinel — replaces it with the
    model's argmax predictions so the explainer attributes wrt. the predicted
    class. Returns the empty dict when no rewriting is needed.

    Accepts a raw classification logits tensor (not the wrapping
    :class:`ForwardOutput`) so the helper stays a pure tensor operation;
    ``assess_transparency`` is responsible for unwrapping.
    """
    raw_config = cfg_to_dict(explainer_config)
    call_config = raw_config.get("call")
    if not isinstance(call_config, dict):
        return {}
    if call_config.get("target") != "auto_pred":
        return {}
    predictions, _ = metrics_prediction_pair(forward_output)
    return {"target": predictions.detach()}


def assess_transparency(
    config: AppConfig,
    model: Model,
    data: Data,
    forward_output: ForwardOutput,
    *,
    input_metadata: InputSpec | None,
    resolved_preprocessing: ResolvedPreprocessing | None = None,
) -> list[ExplanationResult]:
    """Run every explainer declared under ``config.transparency``.

    Returns the explanation results. Each explanation owns its report
    visualisations (``ExplanationResult.visualisations``), populated here via
    ``visualise()``.
    """
    explainers = list((getattr(config, "transparency", None) or {}).items())
    if not explainers:
        return []

    suffix = "s" if len(explainers) > 1 else ""
    raitap_log.info("Performing transparency assessment%s (%d)...", suffix, len(explainers))

    if forward_output.task_kind is TaskKind.detection:
        return _assess_transparency_detection(
            config=config,
            model=model,
            data=data,
            forward_output=forward_output,
            input_metadata=input_metadata,
            resolved_preprocessing=resolved_preprocessing,
            explainer_names=[name for name, _ in explainers],
        )

    # ``auto_pred`` only makes sense for classification logits; for other
    # task kinds we skip the rewriting and let the explainer use its
    # configured target as-is.
    predictions_tensor = (
        forward_output.predictions_tensor
        if forward_output.task_kind is TaskKind.classification
        else None
    )

    explanations: list[ExplanationResult] = []
    for name, _explainer_cfg in explainers:
        runtime_kwargs: dict[str, Any] = {}
        if predictions_tensor is not None:
            runtime_kwargs = resolve_explainer_runtime_kwargs(
                config.transparency[name],
                forward_output=predictions_tensor,
            )
        explanation = Explanation(
            config,
            name,
            model,
            # Detection routed away at the task-kind check above, so this is the
            # dense-NCHW classification path; narrow the ``Data.tensor`` union.
            cast("torch.Tensor", data.tensor),
            input_metadata=input_metadata,
            sample_ids=data.sample_ids,
            sample_names=data.sample_ids,
            resolved_preprocessing=resolved_preprocessing,
            **runtime_kwargs,
        )
        explanation.visualise()  # populates explanation.visualisations
        explanations.append(explanation)
    return explanations


def _assess_transparency_detection(
    *,
    config: AppConfig,
    model: Model,
    data: Data,
    forward_output: ForwardOutput,
    input_metadata: InputSpec | None,
    resolved_preprocessing: ResolvedPreprocessing | None,
    explainer_names: list[str],
) -> list[ExplanationResult]:
    """Detection-task branch — one ExplanationResult per detected box.

    Replicates the per-explainer setup that
    :class:`raitap.transparency.factory.Explanation` performs for
    classification (create explainer + visualisers, compat checks, resolve
    backend + run_dir), then delegates the K-loop to
    :func:`explain_detection`.
    """
    from raitap.configs import resolve_run_dir
    from raitap.configs.adapter_factory import resolve_per_image_transform
    from raitap.pipeline.phases.explain_detection import (
        _DEFAULT_IOU_THRESHOLD,
        explain_detection,
    )
    from raitap.transparency.baselines import apply_config_baseline
    from raitap.transparency.detection_labels import (
        enrich_detection_box,
        resolve_category_names,
    )
    from raitap.transparency.factory import (
        _PARSED_EXPLAINER_CONFIG_CACHE,
        _parse_explainer_config,
        _require_model_backend,
        check_explainer_visualiser_compat,
        check_explainer_visualiser_payload_compat,
        check_explainer_visualiser_semantic_compat,
        create_explainer,
        create_visualisers,
        resolve_call_data_sources,
    )

    explanations: list[ExplanationResult] = []
    backend = _require_model_backend(model)
    category_names = resolve_category_names(
        config.model.class_names,
        backend.category_names,
    )
    data_labels = getattr(data, "labels", None)
    detection_ground_truth = data_labels if isinstance(data_labels, list) else None

    for name in explainer_names:
        explainer_config = config.transparency[name]
        parsed = _parse_explainer_config(explainer_config)
        algorithm = str(parsed.algorithm or "")
        cache_key = id(explainer_config)
        _PARSED_EXPLAINER_CONFIG_CACHE[cache_key] = parsed
        try:
            explainer, explainer_target = create_explainer(explainer_config)
            viz_list = create_visualisers(explainer_config)
            check_explainer_visualiser_compat(explainer_target, algorithm, viz_list)
            check_explainer_visualiser_payload_compat(explainer, explainer_target, viz_list)
            check_explainer_visualiser_semantic_compat(
                explainer,
                explainer_target,
                viz_list,
                task_kind=backend.task_kind,
            )
            explainer.check_backend_compat(backend)

            call_from_config = dict(parsed.call)
            raitap_cfg = dict(parsed.raitap)
            ground_truth_iou_threshold = float(
                raitap_cfg.get("detection", {}).get("iou_threshold", _DEFAULT_IOU_THRESHOLD)
            )
            if data.sample_ids is not None:
                raitap_cfg["sample_ids"] = data.sample_ids
                raitap_cfg["sample_names"] = data.sample_ids
            if input_metadata is not None:
                raitap_cfg["input_metadata"] = input_metadata

            call_from_config = apply_config_baseline(
                explainer=explainer,
                call_kwargs=call_from_config,
                raitap_kwargs=raitap_cfg,
            )

            call_provenance: dict[str, dict[str, Any]] = {}
            merged_kwargs = resolve_call_data_sources(
                call_from_config,
                log_label="call",
                per_image_transform=resolve_per_image_transform(
                    config,
                    resolved_preprocessing=resolved_preprocessing,
                ),
                provenance_out=call_provenance,
            )
            merged_kwargs = backend._prepare_kwargs(merged_kwargs)

            base_run_dir = resolve_run_dir(config, subdir=f"transparency/{name}")

            for result in explain_detection(
                inputs=data.tensor,
                forward_output=forward_output,
                backend=backend,
                explainer=explainer,
                explainer_target=explainer_target,
                explainer_name=name,
                visualisers=viz_list,
                base_run_dir=base_run_dir,
                raitap_kwargs=raitap_cfg,
                call_kwargs=merged_kwargs,
                call_provenance=call_provenance,
            ):
                if result.detection_box is not None:
                    sample_index = result.original_sample_index
                    ground_truth_for_sample = None
                    if detection_ground_truth is not None and sample_index is not None:
                        if sample_index < len(detection_ground_truth):
                            ground_truth_for_sample = detection_ground_truth[sample_index]
                        else:
                            # Loader guarantees len(detection_ground_truth) == n_samples, so this
                            # only fires on a genuine prediction/label misalignment;
                            # surface it instead of silently skipping the GT match.
                            raitap_log.warn(
                                "Detection ground truth has %d entries but an explanation "
                                "references sample_index=%d; rendering this box without a "
                                "true label (prediction/label misalignment).",
                                len(detection_ground_truth),
                                sample_index,
                            )
                    result.detection_box = enrich_detection_box(
                        result.detection_box,
                        category_names=category_names,
                        ground_truth_for_sample=ground_truth_for_sample,
                        iou_threshold=ground_truth_iou_threshold,
                    )
                result.visualise()  # populates result.visualisations
                explanations.append(result)
        finally:
            _PARSED_EXPLAINER_CONFIG_CACHE.pop(cache_key, None)

    return explanations
