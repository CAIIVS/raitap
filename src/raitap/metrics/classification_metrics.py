from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from raitap.configs.schema import (
    BinaryClassificationMetricsConfig,
    MulticlassClassificationMetricsConfig,
    MultilabelClassificationMetricsConfig,
)
from raitap.metrics.registration import metrics_adapter
from raitap.utils.lazy import lazy_import

from .base_metric_computer import BaseMetricComputer, MetricResult
from .utils import tensor_to_python

if TYPE_CHECKING:
    import torch
    import torchmetrics
else:
    # Deferred so this module can be imported in a venv without the
    # ``metrics`` extra installed (e.g. during ``raitap.run(..., auto_install_deps=True)``).
    torchmetrics = lazy_import("torchmetrics")

__all__ = [
    "Average",
    "BinaryClassificationMetrics",
    "MulticlassClassificationMetrics",
    "MultilabelClassificationMetrics",
]

Average = Literal["micro", "macro", "weighted", "none"]


class _ClassificationBase(BaseMetricComputer):
    """Shared update/compute/reset/device wiring; subclasses build the metric quartet."""

    accuracy: torchmetrics.Metric
    precision: torchmetrics.Metric
    recall: torchmetrics.Metric
    f1: torchmetrics.Metric
    average: Average | None

    def _move_to_device(self, device: torch.device | None) -> None:
        if device is None:
            return
        self.accuracy = self.accuracy.to(device)
        self.precision = self.precision.to(device)
        self.recall = self.recall.to(device)
        self.f1 = self.f1.to(device)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        self.accuracy.update(predictions, targets)
        self.precision.update(predictions, targets)
        self.recall.update(predictions, targets)
        self.f1.update(predictions, targets)

    def compute(self) -> MetricResult:
        acc = self.accuracy.compute()
        prec = self.precision.compute()
        rec = self.recall.compute()
        f1 = self.f1.compute()
        metrics: dict[str, float] = {}
        artifacts: dict[str, Any] = {}
        if self.average == "none":
            artifacts["accuracy"] = tensor_to_python(acc)
            artifacts["precision"] = tensor_to_python(prec)
            artifacts["recall"] = tensor_to_python(rec)
            artifacts["f1"] = tensor_to_python(f1)
        else:
            metrics["accuracy"] = float(tensor_to_python(acc))
            metrics["precision"] = float(tensor_to_python(prec))
            metrics["recall"] = float(tensor_to_python(rec))
            metrics["f1"] = float(tensor_to_python(f1))
        return MetricResult(metrics=metrics, artifacts=artifacts)

    def reset(self) -> None:
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

    def _init_quartet(self, **tm_kwargs: Any) -> None:
        self.accuracy = torchmetrics.Accuracy(**tm_kwargs)
        self.precision = torchmetrics.Precision(**tm_kwargs)
        self.recall = torchmetrics.Recall(**tm_kwargs)
        self.f1 = torchmetrics.F1Score(**tm_kwargs)


@metrics_adapter(
    registry_name="binary_classification",
    extra="metrics",
    schema=BinaryClassificationMetricsConfig,
)
class BinaryClassificationMetrics(_ClassificationBase):
    """Binary classification metrics (accuracy, precision, recall, F1)."""

    def __init__(
        self,
        *,
        ignore_index: int | None = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.average = None
        self._init_quartet(
            task="binary",
            ignore_index=ignore_index,
            threshold=threshold,
            **kwargs,
        )


@metrics_adapter(
    registry_name="multiclass_classification",
    extra="metrics",
    schema=MulticlassClassificationMetricsConfig,
)
class MulticlassClassificationMetrics(_ClassificationBase):
    """Multiclass classification metrics (accuracy, precision, recall, F1)."""

    def __init__(
        self,
        *,
        num_classes: int,
        average: Average = "macro",
        ignore_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0.")
        self.average = average
        self._init_quartet(
            task="multiclass",
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )


@metrics_adapter(
    registry_name="multilabel_classification",
    extra="metrics",
    schema=MultilabelClassificationMetricsConfig,
)
class MultilabelClassificationMetrics(_ClassificationBase):
    """Multilabel classification metrics (accuracy, precision, recall, F1)."""

    def __init__(
        self,
        *,
        num_labels: int,
        average: Average = "macro",
        ignore_index: int | None = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        if num_labels <= 0:
            raise ValueError("num_labels must be > 0.")
        self.average = average
        self._init_quartet(
            task="multilabel",
            num_labels=num_labels,
            average=average,
            ignore_index=ignore_index,
            threshold=threshold,
            **kwargs,
        )
