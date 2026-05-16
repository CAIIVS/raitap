from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from raitap.metrics.registration import register_metrics_adapter
from raitap.types import Task
from raitap.utils.lazy import lazy_import

from .base_metric_computer import BaseMetricComputer, MetricResult
from .utils import tensor_to_python

if TYPE_CHECKING:
    import torch
    import torchmetrics
else:
    # Deferred so this module can be imported in a venv without the
    # ``metrics`` extra installed (e.g. during ``raitap.run(..., auto_install=True)``).
    torchmetrics = lazy_import("torchmetrics")

__all__ = ["Average", "ClassificationMetrics", "Task"]

Average = Literal["micro", "macro", "weighted", "none"]


@register_metrics_adapter(registry_name="classification", extra="metrics")
class ClassificationMetrics(BaseMetricComputer):
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
        task: Task | str = Task.multiclass,
        num_classes: int | None = None,
        num_labels: int | None = None,
        average: Average = "macro",
        ignore_index: int | None = None,
        **kwargs: Any,
    ):
        task = Task(task)
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

        self.accuracy = torchmetrics.Accuracy(**tm_task_kwargs, **avg_kwargs)
        self.precision = torchmetrics.Precision(**tm_task_kwargs, **avg_kwargs)
        self.recall = torchmetrics.Recall(**tm_task_kwargs, **avg_kwargs)
        self.f1 = torchmetrics.F1Score(**tm_task_kwargs, **avg_kwargs)

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
