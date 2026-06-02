"""Structured return value from :func:`~raitap.pipeline.run`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, runtime_checkable

from raitap.types import TaskKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from raitap.metrics import MetricsEvaluation
    from raitap.reporting.sections import ReportContext, ReportSection
    from raitap.robustness.results import RobustnessResult, RobustnessVisualisationResult
    from raitap.tracking.base_tracker import BaseTracker
    from raitap.transparency.results import ExplanationResult, VisualisationResult


@runtime_checkable
class PhaseResult(Protocol):
    """An assessment phase's result: loggable to a tracker and renderable to a report.

    Each phase (metrics / transparency / robustness / …) returns one of these
    from its ``run()``; ``RunOutputs.phase_results`` keys them by phase name. The
    pipeline iterates them generically — it never names a concrete phase — so a
    new module plugs in without touching ``pipeline/`` or ``reporting/``.
    """

    report_order: ClassVar[int]

    def log(self, tracker: BaseTracker | None, **kwargs: Any) -> None: ...

    def report_sections(self, ctx: ReportContext) -> Sequence[ReportSection]: ...


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
    """Assessment outputs.

    ``forward_output`` is the typed model forward output; ``phase_results`` holds
    each configured assessment phase's result keyed by phase name (``"metrics"``,
    ``"transparency"``, ``"robustness"``, …). Each value is a :class:`PhaseResult`
    (``Trackable`` + ``Reportable``); the pipeline iterates them generically.
    """

    forward_output: ForwardOutput
    phase_results: dict[str, PhaseResult] = field(default_factory=dict)
    sample_ids: list[str] | None = None
    targets: torch.Tensor | list[dict[str, torch.Tensor]] | None = None
    prediction_summaries: tuple[PredictionSummary, ...] = ()

    # Read-only convenience views over ``phase_results`` for the in-tree phases.
    # ``phase_results`` is the source of truth (and how new modules are reached);
    # these keep the ergonomic ``outputs.metrics`` / ``outputs.explanations``
    # access working. Empty / ``None`` when the phase didn't run.
    @property
    def metrics(self) -> MetricsEvaluation | None:
        return cast("MetricsEvaluation | None", self.phase_results.get("metrics"))

    @property
    def explanations(self) -> list[ExplanationResult]:
        result = self.phase_results.get("transparency")
        return list(getattr(result, "explanations", [])) if result is not None else []

    @property
    def visualisations(self) -> list[VisualisationResult]:
        result = self.phase_results.get("transparency")
        return list(getattr(result, "visualisations", [])) if result is not None else []

    @property
    def robustness_results(self) -> list[RobustnessResult]:
        result = self.phase_results.get("robustness")
        return list(getattr(result, "robustness_results", [])) if result is not None else []

    @property
    def robustness_visualisations(self) -> list[RobustnessVisualisationResult]:
        result = self.phase_results.get("robustness")
        return list(getattr(result, "robustness_visualisations", [])) if result is not None else []
