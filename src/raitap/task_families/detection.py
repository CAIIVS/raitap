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

    def extract_forward(self, ctx: ForwardContext) -> Any:
        raise NotImplementedError

    def explain(self, ctx: ExplainContext) -> list:
        raise NotImplementedError

    def metrics_inputs(self, forward_output: Any, labels: Any) -> Any:
        raise NotImplementedError
