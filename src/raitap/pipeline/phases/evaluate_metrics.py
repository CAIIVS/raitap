"""Metrics phase — evaluates classification metrics when configured."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raitap import raitap_log
from raitap.metrics import (
    Metrics,
    MetricsEvaluation,
    metrics_prediction_pair,
    metrics_run_enabled,
    resolve_metric_targets,
)

if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig


def evaluate_metrics(
    config: AppConfig,
    forward_output: torch.Tensor,
    labels: torch.Tensor | None,
) -> MetricsEvaluation | None:
    """Run metrics on ``forward_output`` if configured; else return ``None``.

    Side-effect: when ``config.metrics.num_classes`` is unset and the forward
    output is shaped like classification logits, the resolved class count is
    written back to the config so downstream consumers see it.
    """
    if not metrics_run_enabled(config):
        return None
    # ``metrics_run_enabled`` already rejected the ``None`` case.
    assert config.metrics is not None

    raitap_log.info("Computing metrics...")
    if (
        getattr(config.metrics, "num_classes", None) is None
        and forward_output.ndim == 2
        and forward_output.shape[1] >= 2
    ):
        config.metrics.num_classes = int(forward_output.shape[1])
    preds, _ = metrics_prediction_pair(forward_output)
    targs = resolve_metric_targets(preds, labels)
    return Metrics(config, preds, targs)
