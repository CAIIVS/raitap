from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from torchmetrics import Accuracy, F1Score, Precision, Recall

from .base_metric import MetricComputer, MetricResult
from .utils import tensor_to_python

if TYPE_CHECKING:
    import torch

    from raitap.tracking.base_tracker import BaseTracker

Task = Literal["binary", "multiclass", "multilabel"]
Average = Literal["micro", "macro", "weighted", "none"]


class ClassificationMetrics(MetricComputer):
    """
    Classification metrics using torchmetrics

    Supports:
        - Task: binary, multiclass, multilabel
        - Average: micro, macro, weighted, none

    Notes:
        - If average="none", metric outputs are per-class/per-label vectors
            and are stored in artifacts.
        - For multilabel, you may want to pass threshold=0.5 (default) in kwargs.
    """

    def __init__(
        self,
        *,
        task: Task = "multiclass",
        num_classes: int | None = None,
        num_labels: int | None = None,
        average: Average = "macro",
        ignore_index: int | None = None,
        **kwargs: Any,
    ):
        if task not in ["binary", "multiclass", "multilabel"]:
            raise ValueError(f"Unknown task '{task}'. Use 'binary', 'multiclass' or 'multilabel'.")

        # Start building arguments dictionary
        tm_task_kwargs: dict[str, Any] = {"task": task, **kwargs}

        if task == "multiclass":
            if num_classes is None or num_classes <= 0:
                raise ValueError("For task='multiclass', you must provide num_classes > 0.")
            tm_task_kwargs["num_classes"] = num_classes
            tm_task_kwargs["ignore_index"] = ignore_index

        elif task == "multilabel":
            # TorchMetrics uses num_labels for multilabel
            if num_labels is None:
                # allow num_classes as alias for num_labels
                if num_classes is None:
                    raise ValueError(
                        "For task='multilabel', provide num_labels (or num_classes as alias)."
                    )
                num_labels = num_classes
            if num_labels <= 0:
                raise ValueError("For task='multilabel', num_labels must be > 0.")
            tm_task_kwargs["num_labels"] = num_labels
            tm_task_kwargs["ignore_index"] = ignore_index
            # If no threshold is provided, use default of 0.5
            tm_task_kwargs.setdefault("threshold", 0.5)
        else:
            # Binary task
            tm_task_kwargs["ignore_index"] = ignore_index

        # Handle average argument
        avg_kwargs: dict[str, Any] = {}
        if task in ("multiclass", "multilabel"):
            avg_kwargs["average"] = average

        self.task = task
        self.average = average

        self.accuracy = Accuracy(**tm_task_kwargs, **avg_kwargs)
        self.precision = Precision(**tm_task_kwargs, **avg_kwargs)
        self.recall = Recall(**tm_task_kwargs, **avg_kwargs)
        self.f1 = F1Score(**tm_task_kwargs, **avg_kwargs)

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

        # When average ist "none", store per-class/per-label metrics in artifacts
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

    def log(self, tracker: BaseTracker) -> None:
        tracker.log_metrics(self.compute().metrics, prefix="performance")
