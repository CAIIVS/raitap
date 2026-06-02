"""Assessment-phase chain — the single source of truth for which deliverable
phases exist, how to tell whether each is configured, and how to run it.

Both the "is any deliverable configured?" guard and the run loop in
:func:`raitap.pipeline.orchestrator.run_without_tracking` iterate
:data:`ASSESSMENT_PHASES`, so neither call site changes when the set of phases
changes. Adding a new assessment module (e.g. fairness) means: add an
:class:`AssessmentPhase` subclass, register it in :data:`ASSESSMENT_PHASES`, and
thread its output onto :class:`PhaseContribution` + :class:`~raitap.pipeline.outputs.RunOutputs`
(the latter is a flat frozen struct, so its fields stay explicit).

Shaped after the Chain-of-Responsibility pattern: each phase independently
decides — via :meth:`AssessmentPhase.is_configured` — whether to contribute.
Unlike classic CoR there is no short-circuit: every configured phase runs and
the pipeline accumulates all contributions, so the chain is a plain ordered list
rather than a ``set_next`` linked list.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from raitap.metrics import metrics_run_enabled
from raitap.pipeline.phases.assess_robustness import assess_robustness
from raitap.pipeline.phases.assess_transparency import assess_transparency
from raitap.pipeline.phases.evaluate_metrics import evaluate_metrics

if TYPE_CHECKING:
    from collections.abc import Iterable

    from raitap.configs.schema import AppConfig
    from raitap.data import Data
    from raitap.data.preprocessing import ResolvedPreprocessing
    from raitap.metrics import MetricsEvaluation
    from raitap.models import Model
    from raitap.pipeline.outputs import ForwardOutput
    from raitap.robustness.results import RobustnessResult, RobustnessVisualisationResult
    from raitap.transparency.contracts import InputSpec
    from raitap.transparency.results import ExplanationResult, VisualisationResult


@dataclass(frozen=True)
class PhaseContext:
    """Everything an assessment phase may need, assembled once per run."""

    config: AppConfig
    model: Model
    data: Data
    forward_output: ForwardOutput
    input_metadata: InputSpec | None
    resolved_preprocessing: ResolvedPreprocessing | None


@dataclass(frozen=True)
class PhaseContribution:
    """A phase's partial contribution to :class:`~raitap.pipeline.outputs.RunOutputs`.

    Each phase fills only the fields it produces; :meth:`merge` folds the
    per-phase contributions into one before the final ``RunOutputs`` is built.
    """

    explanations: list[ExplanationResult] = field(default_factory=list)
    visualisations: list[VisualisationResult] = field(default_factory=list)
    metrics: MetricsEvaluation | None = None
    robustness_results: list[RobustnessResult] = field(default_factory=list)
    robustness_visualisations: list[RobustnessVisualisationResult] = field(default_factory=list)

    @classmethod
    def merge(cls, contributions: Iterable[PhaseContribution]) -> PhaseContribution:
        merged = cls()
        for contribution in contributions:
            metrics = contribution.metrics if contribution.metrics is not None else merged.metrics
            merged = cls(
                explanations=[*merged.explanations, *contribution.explanations],
                visualisations=[*merged.visualisations, *contribution.visualisations],
                metrics=metrics,
                robustness_results=[
                    *merged.robustness_results,
                    *contribution.robustness_results,
                ],
                robustness_visualisations=[
                    *merged.robustness_visualisations,
                    *contribution.robustness_visualisations,
                ],
            )
        return merged


class AssessmentPhase(ABC):
    """One deliverable-producing phase: a configured-check plus a run step."""

    name: ClassVar[str]

    @abstractmethod
    def is_configured(self, config: AppConfig) -> bool:
        """True when this phase has anything to do for ``config`` (config-only)."""

    @abstractmethod
    def run(self, ctx: PhaseContext) -> PhaseContribution:
        """Execute the phase and return its contribution to the run outputs."""


class MetricsPhase(AssessmentPhase):
    name = "metrics"

    def is_configured(self, config: AppConfig) -> bool:
        return metrics_run_enabled(config)

    def run(self, ctx: PhaseContext) -> PhaseContribution:
        return PhaseContribution(
            metrics=evaluate_metrics(ctx.config, ctx.forward_output, ctx.data.labels)
        )


class TransparencyPhase(AssessmentPhase):
    name = "transparency"

    def is_configured(self, config: AppConfig) -> bool:
        return bool(getattr(config, "transparency", None))

    def run(self, ctx: PhaseContext) -> PhaseContribution:
        explanations, visualisations = assess_transparency(
            ctx.config,
            ctx.model,
            ctx.data,
            ctx.forward_output,
            input_metadata=ctx.input_metadata,
            resolved_preprocessing=ctx.resolved_preprocessing,
        )
        return PhaseContribution(explanations=explanations, visualisations=visualisations)


class RobustnessPhase(AssessmentPhase):
    name = "robustness"

    def is_configured(self, config: AppConfig) -> bool:
        return bool(getattr(config, "robustness", None))

    def run(self, ctx: PhaseContext) -> PhaseContribution:
        results, visualisations = assess_robustness(
            ctx.config,
            ctx.model,
            ctx.data,
            ctx.forward_output,
            labels=ctx.data.labels,
            input_metadata=ctx.input_metadata,
            resolved_preprocessing=ctx.resolved_preprocessing,
        )
        return PhaseContribution(
            robustness_results=results,
            robustness_visualisations=visualisations,
        )


# Ordered chain. Order is preserved in execution (metrics -> transparency ->
# robustness, matching the historical orchestrator order).
ASSESSMENT_PHASES: tuple[AssessmentPhase, ...] = (
    MetricsPhase(),
    TransparencyPhase(),
    RobustnessPhase(),
)
