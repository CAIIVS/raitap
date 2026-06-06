"""Transparency assessment phase — instantiates explainers + collects results.

Co-located with the module it drives (issue #243 follow-up): the phase class,
its work function, and runtime-kwarg resolution live here; the result type +
report rendering live in :mod:`raitap.transparency.report`.

``assess_transparency`` is a uniform shell: it resolves the ``TaskFamily`` for
the forward output's kind, runs the shared :func:`prepare_explainer` setup once
per explainer, then delegates the per-kind scope strategy to
``family.explain``. Classification builds one ``ExplanationResult`` per
explainer; detection runs a per-box K-loop. Neither branch lives here anymore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from raitap import raitap_log
from raitap.configs import cfg_to_dict
from raitap.metrics import metrics_prediction_pair
from raitap.pipeline.phases.base import AssessmentPhase
from raitap.task_families import ExplainContext, resolve_task_family
from raitap.transparency.report import TransparencyPhaseResult
from raitap.utils.diagnostics import Module

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import torch

    from raitap.configs.schema import AppConfig
    from raitap.data import Data
    from raitap.data.preprocessing import ResolvedPreprocessing
    from raitap.models import Model
    from raitap.models.backend import ModelBackend
    from raitap.pipeline.outputs import ForwardOutput, PhaseResult
    from raitap.pipeline.phases.base import PhaseContext
    from raitap.transparency.contracts import ExplainerAdapter, InputSpec
    from raitap.transparency.results import ExplanationResult

__all__ = [
    "PreparedExplainer",
    "TransparencyPhase",
    "assess_transparency",
    "prepare_explainer",
    "resolve_explainer_runtime_kwargs",
]


class TransparencyPhase(AssessmentPhase):
    name = "transparency"

    def is_configured(self, config: AppConfig) -> bool:
        return bool(getattr(config, "transparency", None))

    def run(self, ctx: PhaseContext) -> PhaseResult | None:
        explanations = assess_transparency(
            ctx.config,
            ctx.model,
            ctx.data,
            ctx.forward_output,
            input_metadata=ctx.input_metadata,
            resolved_preprocessing=ctx.resolved_preprocessing,
        )
        return TransparencyPhaseResult(explanations=explanations)


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
    ``ClassificationFamily.explain`` is responsible for unwrapping.
    """
    raw_config = cfg_to_dict(explainer_config)
    call_config = raw_config.get("call")
    if not isinstance(call_config, dict):
        return {}
    if call_config.get("target") != "auto_pred":
        return {}
    predictions, _ = metrics_prediction_pair(forward_output)
    return {"target": predictions.detach()}


@dataclass(frozen=True)
class PreparedExplainer:
    """Per-explainer setup shared by every task family's ``explain``.

    ``prepare_explainer`` runs the full config-derived setup once (instantiate
    explainer + visualisers, compat checks, baseline routing, call-data-source
    resolution, backend kwarg prep, run dir). The family-specific scope strategy
    then consumes these frozen fields.

    ``experiment_name``, ``explainer_config`` and ``class_names`` are
    family-specific seams: classification passes ``experiment_name`` to the
    explainer and re-reads ``explainer_config`` to resolve the ``auto_pred``
    runtime target; detection reads ``class_names`` (the configured id->name
    table) to enrich each detected box. Neither family uses the other's seam.
    """

    name: str
    explainer: ExplainerAdapter
    explainer_target: str
    visualisers: list
    merged_kwargs: dict
    raitap_kwargs: dict
    call_provenance: dict
    base_run_dir: Path
    backend: ModelBackend
    experiment_name: str
    explainer_config: object  # consumed via resolve_explainer_runtime_kwargs(Any); stays loose
    class_names: Sequence[str] | None


def prepare_explainer(
    config: AppConfig,
    name: str,
    model: Model,
    *,
    resolved_preprocessing: ResolvedPreprocessing | None,
    input_metadata: InputSpec | None,
    data: Data,
) -> PreparedExplainer:
    """Run the per-explainer setup shared by every task family.

    Instantiates the explainer + visualisers, runs the three compat checks plus
    ``explainer.check_backend_compat``, applies the config baseline, resolves
    call-data-source kwargs (with the per-image transform), prepares the backend
    kwargs, and resolves the run dir. Returns a frozen :class:`PreparedExplainer`.

    This is the single home for the setup that the classification and detection
    paths used to duplicate. The ``auto_pred`` runtime-target rewrite is NOT done
    here — it depends on the forward output and is a classification-only concern,
    so it stays in ``ClassificationFamily.explain``.
    """
    from raitap.configs import resolve_run_dir
    from raitap.configs.adapter_factory import (
        resolve_call_data_sources,
        resolve_per_image_transform,
    )
    from raitap.transparency.baselines import apply_config_baseline
    from raitap.transparency.factory import (
        _PARSED_EXPLAINER_CONFIG_CACHE,
        _parse_explainer_config,
        _require_model_backend,
        check_explainer_visualiser_compat,
        check_explainer_visualiser_payload_compat,
        check_explainer_visualiser_semantic_compat,
        create_explainer,
        create_visualisers,
    )

    backend = _require_model_backend(model)
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
    finally:
        _PARSED_EXPLAINER_CONFIG_CACHE.pop(cache_key, None)

    return PreparedExplainer(
        name=name,
        explainer=explainer,
        explainer_target=explainer_target,
        visualisers=viz_list,
        merged_kwargs=merged_kwargs,
        raitap_kwargs=raitap_cfg,
        call_provenance=call_provenance,
        base_run_dir=base_run_dir,
        backend=backend,
        experiment_name=str(getattr(config, "experiment_name", "")),
        explainer_config=explainer_config,
        # ``config.model`` is a struct-mode DictConfig; when the YAML omits the
        # optional ``class_names`` key an unconditional read raises
        # ``ConfigAttributeError``. Read defensively so the optional field +
        # backend fallback in ``resolve_category_names`` work as designed (#240).
        class_names=getattr(config.model, "class_names", None),
    )


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

    Resolves the ``TaskFamily`` for the forward output's kind, runs the shared
    :func:`prepare_explainer` setup once per explainer, then delegates the
    per-kind scope strategy to ``family.explain``. Each explanation owns its
    report visualisations (``ExplanationResult.visualisations``).
    """
    explainer_names = list((getattr(config, "transparency", None) or {}).keys())
    if not explainer_names:
        return []

    suffix = "s" if len(explainer_names) > 1 else ""
    raitap_log.info(
        "Performing transparency assessment%s (%d)...",
        suffix,
        len(explainer_names),
        module=Module["transparency"],
    )

    family = resolve_task_family(forward_output.task_kind)
    results: list[ExplanationResult] = []
    for name in explainer_names:
        prepared = prepare_explainer(
            config,
            name,
            model,
            resolved_preprocessing=resolved_preprocessing,
            input_metadata=input_metadata,
            data=data,
        )
        results += family.explain(
            ExplainContext(prepared=prepared, forward_output=forward_output, data=data)
        )
    return results
