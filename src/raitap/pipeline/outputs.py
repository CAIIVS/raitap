"""Structured return value from :func:`~raitap.pipeline.run`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from raitap.types import TaskKind

if TYPE_CHECKING:
    import torch

    from raitap.metrics import MetricsEvaluation
    from raitap.robustness.results import RobustnessResult, RobustnessVisualisationResult
    from raitap.transparency.results import ExplanationResult, VisualisationResult


@dataclass(frozen=True)
class PredictionSummary:
    """Per-sample classification summary used by reporting."""

    sample_index: int
    predicted_class: int
    confidence: float
    sample_id: str | None = None
    target_class: int | None = None
    correct: bool | None = None


@dataclass(frozen=True)
class ForwardOutput:
    """Typed model forward output.

    Replaces the historical ``RunOutputs.forward_output: torch.Tensor`` so
    detection backends (whose forward produces ``list[dict[str, Tensor]]``)
    plug into the same downstream phases without overloading the tensor
    field. Classification path keeps the original tensor shape on
    :attr:`predictions_tensor`; detection path populates
    :attr:`detection_predictions`. :attr:`batch_size` is task-agnostic so
    reporting + UI callers don't need to branch.
    """

    task_kind: TaskKind
    batch_size: int
    predictions_tensor: torch.Tensor | None = None
    detection_predictions: list[dict[str, torch.Tensor]] | None = None

    def __post_init__(self) -> None:
        if self.task_kind is TaskKind.classification and self.predictions_tensor is None:
            raise ValueError("ForwardOutput(task_kind=classification) requires predictions_tensor.")
        if self.task_kind is TaskKind.detection and self.detection_predictions is None:
            raise ValueError("ForwardOutput(task_kind=detection) requires detection_predictions.")


@dataclass(frozen=True)
class RunOutputs:
    """Assessment outputs; ``forward_output`` is the typed model forward output."""

    explanations: list[ExplanationResult]
    visualisations: list[VisualisationResult]
    metrics: MetricsEvaluation | None
    forward_output: ForwardOutput
    sample_ids: list[str] | None = None
    targets: torch.Tensor | list[dict[str, torch.Tensor]] | None = None
    prediction_summaries: tuple[PredictionSummary, ...] = ()
    robustness_results: list[RobustnessResult] = field(default_factory=list)
    robustness_visualisations: list[RobustnessVisualisationResult] = field(default_factory=list)
