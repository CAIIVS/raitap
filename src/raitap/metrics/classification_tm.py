from __future__ import annotations

from typing import Any, Literal

import torch
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall

from raitap.metrics.base import MetricComputer, MetricResult


def _as_float(x: Any) -> float:
    try:
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot convert '{x}' to float: {e}")


class ClassificationMetrics(MetricComputer):
    """
    Classification metrics using torchmetrics
    """

    def __init__(
        self,
        *,
        num_classes: int,
        task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
        average: Literal["micro", "macro", "weighted"] = "macro",
        ignore_index: int | None = None,
        **kwargs: Any,
    ):
        if task not in ["binary", "multiclass", "multilabel"]:
            raise ValueError(f"Unknown task '{task}'. Use 'binary', 'multiclass' or 'multilabel'.")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
        if average not in ["micro", "macro", "weighted"]:
            raise ValueError(f"Unknown average '{average}'. Use 'micro', 'macro' or 'weighted'.")

        self.accuracy = Accuracy(
            task=task, num_classes=num_classes, average=average, ignore_index=ignore_index, **kwargs
        )
        self.precision = Precision(
            task=task, num_classes=num_classes, average=average, ignore_index=ignore_index, **kwargs
        )
        self.recall = Recall(
            task=task, num_classes=num_classes, average=average, ignore_index=ignore_index, **kwargs
        )
        self.f1 = F1Score(
            task=task, num_classes=num_classes, average=average, ignore_index=ignore_index, **kwargs
        )

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        self.accuracy.update(predictions, targets)
        self.precision.update(predictions, targets)
        self.recall.update(predictions, targets)
        self.f1.update(predictions, targets)

    def compute(self) -> MetricResult:
        return MetricResult(
            metrics={
                "accuracy": _as_float(self.accuracy.compute()),
                "precision": _as_float(self.precision.compute()),
                "recall": _as_float(self.recall.compute()),
                "f1": _as_float(self.f1.compute()),
            }
        )

    def reset(self) -> None:
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
