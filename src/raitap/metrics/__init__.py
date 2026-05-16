"""
RAITAP Metrics Module

Provides performance metric computation for classification and detection tasks
using torchmetrics.

Metrics Public Surface
----------------------
MetricComputer
    Protocol defining the interface for all metric computers (reset, update, compute).

MetricResult
    Dataclass containing computed metrics (dict[str, float]) and artifacts (dict[str, Any]).

Metric classes
--------------
ClassificationMetrics
    Computes accuracy, precision, recall, and F1 score for binary, multiclass,
    and multilabel classification tasks.

DetectionMetrics
    Computes mean average precision (mAP) and related metrics for object detection tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Base protocol and result type
from .base_metric_computer import BaseMetricComputer, MetricResult, scalar_metrics_for_tracking

# Concrete metric implementations
from .classification_metrics import ClassificationMetrics
from .detection_metrics import DetectionMetrics
from .factory import (
    Metrics,
    MetricsEvaluation,
    create_metric,
    evaluate,
    metrics_run_enabled,
)
from .inputs import metrics_prediction_pair, resolve_metric_targets
from .visualizers import MetricsVisualizer

if TYPE_CHECKING:
    from raitap.configs.schema import MetricsConfig
    from raitap.types import Task


def __getattr__(name: str) -> Any:
    """Resolve hydra-zen builders by registry name, plus the schema dataclass
    (:class:`~raitap.configs.schema.MetricsConfig`) and the
    :class:`~raitap.types.Task` enum re-exported here so the module owns
    both the type contract and the builder instances."""
    if name == "MetricsConfig":
        from raitap.configs.schema import MetricsConfig

        return MetricsConfig
    if name == "Task":
        from raitap.types import Task

        return Task
    from raitap._adapters import lookup

    return lookup("metrics", name)


__all__ = [  # noqa: RUF022
    # Schema dataclass + task enum (lazy)
    "MetricsConfig",
    "Task",
    # Base types
    "BaseMetricComputer",
    "MetricResult",
    "scalar_metrics_for_tracking",
    # Concrete implementations
    "ClassificationMetrics",
    "DetectionMetrics",
    # Factory API
    "Metrics",
    "MetricsEvaluation",
    "create_metric",
    "evaluate",
    "metrics_prediction_pair",
    "metrics_run_enabled",
    "resolve_metric_targets",
    # Visualization
    "MetricsVisualizer",
]
