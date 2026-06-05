"""Detection task family.

Per-box explanation semantics (K boxes per sample) live here. Robustness is a
Phase 4 deliverable, so ``supports_robustness`` is ``False``; detection models
normalise internally, so ``allows_preprocessing`` is ``False``. Heavy members
are added in the phase-migration tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.task_families.registry import task_family
from raitap.transparency.contracts import ExplanationOutputSpace
from raitap.types import TaskKind

if TYPE_CHECKING:
    from raitap.task_families.base import ExplainContext, ForwardContext


@task_family
class DetectionFamily:
    kind: TaskKind = TaskKind.detection
    output_space: ExplanationOutputSpace = ExplanationOutputSpace.DETECTION_BOXES

    def validate_payload(self, payload: object) -> None:
        if not isinstance(payload, list) or not all(isinstance(p, dict) for p in payload):
            raise ValueError("ForwardOutput(task_kind=detection) requires a list[dict] payload.")

    def supports_robustness(self) -> bool:
        return False

    @property
    def allows_preprocessing(self) -> bool:
        return False

    def prediction_summaries(self, payload: Any) -> list | None:
        # Detection has no per-sample "predicted class + confidence" concept.
        return None

    # Heavy members filled in by later phase-migration tasks.
    def load_inputs(self, cfg: Any) -> Any:
        raise NotImplementedError

    def load_labels(self, cfg: Any) -> Any:
        raise NotImplementedError

    def extract_forward(self, ctx: ForwardContext, *, batch_size: int) -> list[dict]:
        import torch

        backend, inputs = ctx.backend, ctx.inputs
        total_batch = len(inputs)
        detection_predictions: list[dict] = []
        for start in range(0, total_batch, batch_size):
            end = min(start + batch_size, total_batch)
            prepared = backend.prepare_detection_inputs(inputs[start:end])
            raw = backend(prepared)
            if not isinstance(raw, list):
                raise TypeError(
                    "forward_pass(detection) expected list[dict] from backend; "
                    f"got {type(raw).__name__}."
                )
            for sample_dict in raw:
                if not isinstance(sample_dict, dict):
                    raise TypeError(
                        "forward_pass(detection) expected each backend output entry to be a "
                        f"dict of tensors; got {type(sample_dict).__name__}."
                    )
                detection_predictions.append({k: v.detach().cpu() for k, v in sample_dict.items()})
            del prepared, raw
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return detection_predictions

    def explain(self, ctx: ExplainContext) -> list:
        """Yield one ``ExplanationResult`` per detected box (K-loop).

        Resolves the id->name table once (explicit config > backend), drives
        :func:`explain_detection` over the pre-computed predictions, and enriches
        each raw box with the predicted name + (optional) ground-truth match
        before populating its report visualisations.
        """
        from raitap import raitap_log
        from raitap.transparency.detection_labels import (
            enrich_detection_box,
            resolve_category_names,
        )
        from raitap.transparency.explain_detection import (
            _DEFAULT_IOU_THRESHOLD,
            explain_detection,
        )

        prepared = ctx.prepared
        backend = prepared.backend

        category_names = resolve_category_names(
            prepared.class_names,
            backend.category_names,
        )
        data_labels = getattr(ctx.data, "labels", None)
        detection_ground_truth = data_labels if isinstance(data_labels, list) else None

        raitap_cfg = prepared.raitap_kwargs
        ground_truth_iou_threshold = float(
            raitap_cfg.get("detection", {}).get("iou_threshold", _DEFAULT_IOU_THRESHOLD)
        )

        explanations: list = []
        for result in explain_detection(
            inputs=ctx.data.tensor,
            forward_output=ctx.forward_output,
            backend=backend,
            explainer=prepared.explainer,
            explainer_target=prepared.explainer_target,
            explainer_name=prepared.name,
            visualisers=prepared.visualisers,
            base_run_dir=prepared.base_run_dir,
            raitap_kwargs=raitap_cfg,
            call_kwargs=prepared.merged_kwargs,
            call_provenance=prepared.call_provenance,
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
            result._visualise()  # populates result.visualisations
            explanations.append(result)
        return explanations

    def metrics_inputs(self, forward_output: Any, labels: Any) -> Any:
        raise NotImplementedError
