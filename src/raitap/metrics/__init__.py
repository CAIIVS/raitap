"""
RAITAP Metrics Module

Provides performance metric computation for classification and detection tasks
using torchmetrics.

Public API
----------
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

# Base protocol and result type
from .base_metric import BaseMetricComputer, MetricResult, scalar_metrics_for_tracking

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

__all__ = [  # noqa: RUF022
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
]
