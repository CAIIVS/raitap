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
from .base import MetricComputer, MetricResult

# Concrete metric implementations
from .classification_tm import ClassificationMetrics
from .detection_tm import DetectionMetrics

__all__ = [
    # Base types
    "MetricComputer",
    "MetricResult",
    # Concrete implementations
    "ClassificationMetrics",
    "DetectionMetrics",
]
