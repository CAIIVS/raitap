"""Classification task family.

Classification is a peer family, not the default (spec D4). The dense-NCHW
input path, the logits-tensor ``payload``, and shared-loop transparency all
live here. Heavy members (load/forward/explain/metrics) are added in the
phase-migration tasks; this file starts with the constants + validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from raitap.task_families.registry import task_family
from raitap.transparency.contracts import ExplanationOutputSpace
from raitap.types import TaskKind

if TYPE_CHECKING:
    from raitap.task_families.base import ExplainContext, ForwardContext


@task_family
class ClassificationFamily:
    kind: TaskKind = TaskKind.classification
    output_space: ExplanationOutputSpace = ExplanationOutputSpace.INPUT_FEATURES

    def validate_payload(self, payload: object) -> None:
        if not isinstance(payload, torch.Tensor):
            raise ValueError("ForwardOutput(task_kind=classification) requires a tensor payload.")

    def supports_robustness(self) -> bool:
        return True

    @property
    def allows_preprocessing(self) -> bool:
        return True

    # Heavy members filled in by later phase-migration tasks.
    def load_inputs(self, cfg: Any) -> Any:  # Task 8
        raise NotImplementedError

    def load_labels(self, cfg: Any) -> Any:  # Task 8
        raise NotImplementedError

    def extract_forward(self, ctx: ForwardContext) -> Any:  # Task 6
        raise NotImplementedError

    def explain(self, ctx: ExplainContext) -> list:  # Task 7
        raise NotImplementedError

    def metrics_inputs(self, forward_output: Any, labels: Any) -> Any:  # Task 9
        raise NotImplementedError

    def prediction_summaries(self, payload: Any) -> list | None:  # Task 11
        raise NotImplementedError
