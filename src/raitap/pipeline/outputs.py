"""Structured return value from :func:`~raitap.pipeline.run`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, runtime_checkable

from raitap.types import TaskKind

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import torch

    from raitap.metrics import MetricsEvaluation
    from raitap.metrics.base_metric_computer import MetricResult
    from raitap.reporting.sections import ReportContext, ReportSection
    from raitap.robustness.results import RobustnessResult
    from raitap.tracking.base_tracker import BaseTracker, Trackable
    from raitap.transparency.results import ExplanationResult


@runtime_checkable
class PhaseResult(Protocol):
    """An assessment phase's result: loggable to a tracker and renderable to a report.

    Each phase (metrics / transparency / robustness / …) returns one of these
    from its ``run()``; ``RunOutputs.phase_results`` keys them by phase name. The
    pipeline iterates them generically — it never names a concrete phase — so a
    new module plugs in without touching ``pipeline/`` or ``reporting/``.
    """

    report_order: ClassVar[int]

    def log(self, tracker: BaseTracker | None, **kwargs: Any) -> None:
        """Log this phase's artifacts / metrics to the tracker."""
        raise NotImplementedError

    def report_sections(self, ctx: ReportContext) -> Sequence[ReportSection]:
        """Return this phase's contribution as ordered report sections."""
        raise NotImplementedError


@runtime_checkable
class AdapterResult(Protocol):
    """Common envelope every per-adapter assessment result implements.

    Identity (config key / library class / method), provenance (``run_dir``),
    the figures it owns (1:N), and — load-bearing for RAITAP's semantic-
    transparency thesis — its ``semantics``. The domain *payload* (attributions
    vs adversarial tensors vs …) is deliberately NOT part of this contract; it
    differs per module. Metrics is a singleton (no adapter loop, no figures) and
    does NOT implement it.
    """

    # Read-only properties (not bare attributes) so the members are covariant —
    # a concrete result's ``semantics: ExplanationSemantics`` / ``visualisations:
    # list[VisualisationResult]`` satisfy ``object`` / ``Sequence[Trackable]``.
    # Dataclass fields satisfy a read-only protocol property.
    @property
    def name(self) -> str | None: ...
    @property
    def adapter_target(self) -> str: ...
    @property
    def algorithm(self) -> str: ...
    @property
    def semantics(self) -> object: ...
    @property
    def run_dir(self) -> Path: ...
    @property
    def visualisations(self) -> Sequence[Trackable]: ...


class _RenderableResult(AdapterResult, Protocol):  # noqa: PYI046  # used as the run_adapters TypeVar bound (TYPE_CHECKING string ref)
    """Internal: what ``run_adapters`` consumes — the envelope plus the
    persist-and-render step it drives. ``_visualise`` is underscored because it
    is pipeline-internal (side-effecting: writes PNGs, mutates state)."""

    def _visualise(self, **kwargs: Any) -> Sequence[Trackable]: ...


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

    # Generic, phase-agnostic mapping access — for any phase, incl. future
    # modules: ``outputs.get("fairness")`` / ``outputs["metrics"]`` / ``"x" in outputs``.
    def get(self, phase: str, default: PhaseResult | None = None) -> PhaseResult | None:
        return self.phase_results.get(phase, default)

    def __getitem__(self, phase: str) -> PhaseResult:
        return self.phase_results[phase]

    def __contains__(self, phase: object) -> bool:
        return phase in self.phase_results

    # --- Typed convenience views (THE per-phase coupling point) -------------
    # Ergonomic, statically-typed access to each in-tree phase's result(s):
    # ``outputs.<phase>`` returns the result data (a list for the per-adapter
    # families, the single ``MetricResult`` for metrics). Each per-adapter
    # result owns its ``.visualisations``. ``phase_results`` stays the source of
    # truth; these only read from it. Future/dynamic phases: ``outputs.get(...)``.
    @property
    def metrics(self) -> MetricResult | None:
        evaluation = cast("MetricsEvaluation | None", self.phase_results.get("metrics"))
        return evaluation.result if evaluation is not None else None

    @property
    def transparency(self) -> list[ExplanationResult]:
        result = self.phase_results.get("transparency")
        return list(getattr(result, "explanations", []))

    @property
    def robustness(self) -> list[RobustnessResult]:
        result = self.phase_results.get("robustness")
        return list(getattr(result, "results", []))
