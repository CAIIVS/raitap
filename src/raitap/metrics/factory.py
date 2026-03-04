"""Factory utilities for metrics instantiation."""

from __future__ import annotations

from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from .base import MetricComputer


def create_metric(config: DictConfig | dict[str, Any]) -> MetricComputer:
    """
    Instantiate a metric computer from a Hydra config.

    Args:
        config: Configuration dict with ``_target_`` key pointing to a
                MetricComputer implementation (e.g., ClassificationMetrics,
                DetectionMetrics).

    Returns:
        Instantiated metric computer.

    Example:
        >>> from omegaconf import DictConfig
        >>> cfg = DictConfig({
        ...     "_target_": "raitap.metrics.ClassificationMetrics",
        ...     "task": "multiclass",
        ...     "num_classes": 10,
        ...     "average": "macro"
        ... })
        >>> metric = create_metric(cfg)
    """
    return instantiate(config)
