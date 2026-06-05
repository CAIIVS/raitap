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
        raise NotImplementedError

    def metrics_inputs(self, forward_output: Any, labels: Any) -> Any:
        raise NotImplementedError
