"""Assessment-phase chain — the single source of truth for which deliverable
phases exist, how to tell whether each is configured, and how to run it.

Both the "is any deliverable configured?" guard and the run loop in
:func:`raitap.pipeline.orchestrator.run_without_tracking` iterate
:data:`ASSESSMENT_PHASES`, so neither call site changes when the set of phases
changes. Adding a new assessment module (e.g. fairness) means: add an
:class:`AssessmentPhase` subclass whose ``run`` returns a result implementing
:class:`~raitap.pipeline.outputs.PhaseResult` (``Trackable`` + ``Reportable``),
and register it in :data:`ASSESSMENT_PHASES`. The orchestrator stores the result
under ``RunOutputs.phase_results[name]`` and the tracker loop + report builder
dispatch over it generically — no edits to ``pipeline/`` or ``reporting/``.

Shaped after the Chain-of-Responsibility pattern: each phase independently
decides — via :meth:`AssessmentPhase.is_configured` — whether to contribute.
Unlike classic CoR there is no short-circuit: every configured phase runs and
the pipeline accumulates all results, so the chain is a plain ordered list
rather than a ``set_next`` linked list.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from raitap.metrics import metrics_run_enabled
from raitap.pipeline.phases.assess_robustness import assess_robustness
from raitap.pipeline.phases.assess_transparency import assess_transparency
from raitap.pipeline.phases.evaluate_metrics import evaluate_metrics
from raitap.robustness.report import RobustnessPhaseResult
from raitap.transparency.report import TransparencyPhaseResult

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.data import Data
    from raitap.data.preprocessing import ResolvedPreprocessing
    from raitap.models import Model
    from raitap.pipeline.outputs import ForwardOutput, PhaseResult
    from raitap.transparency.contracts import InputSpec


@dataclass(frozen=True)
class PhaseContext:
    """Everything an assessment phase may need, assembled once per run."""

    config: AppConfig
    model: Model
    data: Data
    forward_output: ForwardOutput
    input_metadata: InputSpec | None
    resolved_preprocessing: ResolvedPreprocessing | None


class AssessmentPhase(ABC):
    """One deliverable-producing phase: a configured-check plus a run step."""

    name: ClassVar[str]

    @abstractmethod
    def is_configured(self, config: AppConfig) -> bool:
        """True when this phase has anything to do for ``config`` (config-only)."""

    @abstractmethod
    def run(self, ctx: PhaseContext) -> PhaseResult | None:
        """Run the phase; return its result, or ``None`` when it produced nothing."""


class MetricsPhase(AssessmentPhase):
    name = "metrics"

    def is_configured(self, config: AppConfig) -> bool:
        return metrics_run_enabled(config)

    def run(self, ctx: PhaseContext) -> PhaseResult | None:
        # ``MetricsEvaluation`` is itself a PhaseResult; ``None`` when metrics ran
        # but produced nothing (e.g. detection labels missing).
        return evaluate_metrics(ctx.config, ctx.forward_output, ctx.data.labels)


class TransparencyPhase(AssessmentPhase):
    name = "transparency"

    def is_configured(self, config: AppConfig) -> bool:
        return bool(getattr(config, "transparency", None))

    def run(self, ctx: PhaseContext) -> PhaseResult | None:
        explanations, visualisations = assess_transparency(
            ctx.config,
            ctx.model,
            ctx.data,
            ctx.forward_output,
            input_metadata=ctx.input_metadata,
            resolved_preprocessing=ctx.resolved_preprocessing,
        )
        return TransparencyPhaseResult(explanations=explanations, visualisations=visualisations)


class RobustnessPhase(AssessmentPhase):
    name = "robustness"

    def is_configured(self, config: AppConfig) -> bool:
        return bool(getattr(config, "robustness", None))

    def run(self, ctx: PhaseContext) -> PhaseResult | None:
        results, visualisations = assess_robustness(
            ctx.config,
            ctx.model,
            ctx.data,
            ctx.forward_output,
            labels=ctx.data.labels,
            input_metadata=ctx.input_metadata,
            resolved_preprocessing=ctx.resolved_preprocessing,
        )
        return RobustnessPhaseResult(
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
