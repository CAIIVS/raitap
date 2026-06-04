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
    metrics_prediction_pair,
    metrics_run_enabled,
    resolve_metric_targets,
)
from raitap.pipeline.phases.base import AssessmentPhase
from raitap.types import TaskKind

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

    Classification path reads ``forward_output.as_classification()`` and threads
    it through ``metrics_prediction_pair`` + ``resolve_metric_targets`` (the
    existing fallback-to-argmax pipeline). Detection path passes
    ``forward_output.as_detection()`` straight to the configured metric
    adapter (e.g. ``DetectionMetrics``); targets must be a ``list[dict]`` from
    the D22 detection label loader.

    Side-effect: when ``config.metrics.num_classes`` is unset and the
    classification forward output is shaped like logits, the resolved class
    count is written back to the config so downstream consumers see it.
    """
    if not metrics_run_enabled(config):
        return None
    # ``metrics_run_enabled`` already rejected the ``None`` case.
    assert config.metrics is not None

    raitap_log.info("Computing metrics...")

    if forward_output.task_kind is TaskKind.detection:
        detection_predictions = forward_output.as_detection()
        if labels is None:
            raitap_log.warn(
                "Detection metrics require dataset labels "
                "(list[dict] from data.labels.kind=detection); none provided. "
                "Skipping metrics."
            )
            return None
        if not isinstance(labels, list):
            raitap_log.warn(
                "Detection metrics require list[dict] targets; got "
                f"{type(labels).__name__}. Skipping metrics."
            )
            return None
        return Metrics(config, detection_predictions, labels)

    if forward_output.task_kind is not TaskKind.classification:
        raitap_log.warn(
            f"Metrics for task_kind={forward_output.task_kind!r} are not wired; skipping."
        )
        return None

    predictions_tensor = forward_output.as_classification()
    if (
        getattr(config.metrics, "num_classes", None) is None
        and predictions_tensor.ndim == 2
        and predictions_tensor.shape[1] >= 2
    ):
        # ``MetricsConfig`` base only carries ``_target_``; ``num_classes``
        # lives on the multiclass typed subclass at runtime.
        config.metrics.num_classes = int(predictions_tensor.shape[1])  # type: ignore[attr-defined]
    preds, _ = metrics_prediction_pair(predictions_tensor)
    classification_labels = labels if not isinstance(labels, list) else None
    targs = resolve_metric_targets(preds, classification_labels)
    return Metrics(config, preds, targs)
