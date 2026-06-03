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
BinaryClassificationMetrics
    Accuracy, precision, recall, and F1 for two-class problems.

MulticlassClassificationMetrics
    Accuracy, precision, recall, and F1 for multiclass classification.

MultilabelClassificationMetrics
    Accuracy, precision, recall, and F1 for multilabel classification.

DetectionMetrics
    Computes mean average precision (mAP) and related metrics for object detection tasks.

Module layout (for contributors):

- ``phase.py`` — pipeline entry point: ``MetricsPhase`` (what the registry assembles) + the ``evaluate_metrics`` work fn. Singleton phase (no adapter loop). Start here to follow a run.
- ``factory.py`` — ``metrics_run_enabled`` + ``evaluate`` (instantiation).
- ``*_metrics.py`` — the metric adapters (classification, detection).
- ``inputs.py`` — target / prediction alignment + label fallbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Base protocol and result type
from .base_metric_computer import BaseMetricComputer, MetricResult, scalar_metrics_for_tracking

# Concrete metric implementations
from .classification_metrics import (
    BinaryClassificationMetrics,
    MulticlassClassificationMetrics,
    MultilabelClassificationMetrics,
)
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


def __getattr__(name: str) -> Any:
    """Resolve hydra-zen builders by registry name, plus the schema dataclass
    (:class:`~raitap.configs.schema.MetricsConfig`) re-exported here so the
    module owns both the type contract and the builder instances."""
    if name == "MetricsConfig":
        from raitap.configs.schema import MetricsConfig

        return MetricsConfig
    from raitap._adapters import lookup

    try:
        return lookup("metrics", name)
    except AttributeError:
        from raitap.configs import register_configs

        register_configs()  # idempotent; fires in-tree imports + plugin discovery
        return lookup("metrics", name)


__all__ = [  # noqa: RUF022
    # Schema dataclass (lazy)
    "MetricsConfig",
    # Base types
    "BaseMetricComputer",
    "MetricResult",
    "scalar_metrics_for_tracking",
    # Concrete implementations
    "BinaryClassificationMetrics",
    "MulticlassClassificationMetrics",
    "MultilabelClassificationMetrics",
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
