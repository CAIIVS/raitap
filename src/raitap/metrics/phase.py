"""Metrics assessment phase.

Co-located with the metrics module: the phase decides whether metrics are
configured and runs them, returning the ``MetricsEvaluation`` (itself the
phase's ``PhaseResult``). Registered in
:data:`raitap.pipeline.phases.registry.ASSESSMENT_PHASES`.

Metrics is a *singleton* phase — a single evaluation, no per-adapter loop and no
``visualise()`` — so it does not use the shared ``run_adapters`` helper that the
transparency/robustness phases share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raitap import raitap_log
from raitap.metrics import (
    Metrics,
    metrics_run_enabled,
)
from raitap.pipeline.phases.base import AssessmentPhase
from raitap.task_families import resolve_task_family

if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig
    from raitap.metrics import MetricsEvaluation
    from raitap.pipeline.outputs import ForwardOutput, PhaseResult
    from raitap.pipeline.phases.base import PhaseContext


class MetricsPhase(AssessmentPhase):
    name = "metrics"

    def is_configured(self, config: AppConfig) -> bool:
        return metrics_run_enabled(config)

    def run(self, ctx: PhaseContext) -> PhaseResult | None:
        # ``MetricsEvaluation`` is itself a PhaseResult; ``None`` when metrics ran
        # but produced nothing (e.g. detection labels missing).
        return evaluate_metrics(ctx.config, ctx.forward_output, ctx.data.labels)


def evaluate_metrics(
    config: AppConfig,
    forward_output: ForwardOutput,
    labels: torch.Tensor | list[dict[str, torch.Tensor]] | None,
) -> MetricsEvaluation | None:
    """Run metrics on ``forward_output`` if configured; else return ``None``.

    Resolves the task family for ``forward_output.task_kind`` and delegates the
    per-kind ``(preds, targets)`` adaptation to ``TaskFamily.metrics_inputs``,
    which may return ``None`` to skip metrics (e.g. detection labels missing).
    """
    if not metrics_run_enabled(config):
        return None
    # ``metrics_run_enabled`` already rejected the ``None`` case.
    assert config.metrics is not None

    raitap_log.info("Computing metrics...")

    family = resolve_task_family(forward_output.task_kind)
    pair = family.metrics_inputs(config, forward_output, labels)
    if pair is None:
        return None
    preds, targs = pair
    return Metrics(config, preds, targs)
