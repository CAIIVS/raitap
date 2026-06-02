"""Metrics assessment phase.

Co-located with the metrics module: the phase decides whether metrics are
configured and runs them, returning the ``MetricsEvaluation`` (itself the
phase's ``PhaseResult``). Registered in
:data:`raitap.pipeline.phases.registry.ASSESSMENT_PHASES`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raitap.metrics.factory import metrics_run_enabled
from raitap.pipeline.phases.base import AssessmentPhase
from raitap.pipeline.phases.evaluate_metrics import evaluate_metrics

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.pipeline.outputs import PhaseResult
    from raitap.pipeline.phases.base import PhaseContext


class MetricsPhase(AssessmentPhase):
    name = "metrics"

    def is_configured(self, config: AppConfig) -> bool:
        return metrics_run_enabled(config)

    def run(self, ctx: PhaseContext) -> PhaseResult | None:
        # ``MetricsEvaluation`` is itself a PhaseResult; ``None`` when metrics ran
        # but produced nothing (e.g. detection labels missing).
        return evaluate_metrics(ctx.config, ctx.forward_output, ctx.data.labels)
