from __future__ import annotations

from typing import Any, Literal

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .base_metric import BaseMetricComputer, MetricResult
from .utils import tensor_to_python

BoxFormat = Literal["xyxy", "xywh"]  # torchvision outputs xyxy
IoUType = Literal["bbox", "segm"] | tuple[Literal["bbox", "segm"], ...]
Backend = Literal["pycocotools", "faster_coco_eval"]
Average = Literal["macro", "micro"]


class DetectionMetrics(BaseMetricComputer):
    """
    Calculates and manages detection metrics for evaluating
    the performance of object detection models.

    This class is responsible for computing detection metrics,
    updating them with predictions and targets, and resetting their state.
    It uses a MeanAveragePrecision calculator internally to handle
    the computation logic. It supports a variety of configurations,
    including box formats, IoU types, thresholds, class-specific metrics, and more.

    :ivar metric: Instance of the MeanAveragePrecision calculator used to compute metrics.
    :type metric: MeanAveragePrecision
    """

    def __init__(
        self,
        *,
        box_format: BoxFormat = "xyxy",
        iou_type: IoUType = "bbox",
        iou_thresholds: list[float] | None = None,
        rec_thresholds: list[float] | None = None,
        max_detection_thresholds: list[int] | None = None,
        class_metrics: bool = False,
        extended_summary: bool = False,
        average: Average = "macro",
        backend: Backend = "faster_coco_eval",
        **kwargs: Any,
    ):
        self.metric = MeanAveragePrecision(
            box_format=box_format,
            iou_type=iou_type,
            iou_thresholds=iou_thresholds,
            rec_thresholds=rec_thresholds,
            max_detection_thresholds=max_detection_thresholds,
            class_metrics=class_metrics,
            extended_summary=extended_summary,
            average=average,
            backend=backend,
            **kwargs,
        )

    def _move_to_device(self, device: torch.device | None) -> None:
        if device is None:
            return
        self.metric = self.metric.to(device)

    def update(self, predictions: Any, targets: Any) -> None:
        # Sanity checks
        if not isinstance(predictions, list) or not isinstance(targets, list):
            raise TypeError(
                f"Expected lists of predictions and targets, "
                f"got {type(predictions)} and {type(targets)}"
            )
        if len(predictions) != len(targets):
            raise ValueError(
                f"Predictions and targets must have the same length, "
                f"got {len(predictions)} and {len(targets)}"
            )

        self.metric.update(predictions, targets)

    def compute(self) -> MetricResult:
        out: dict[str, Any] = self.metric.compute()

        metrics: dict[str, float] = {}
        artifacts: dict[str, Any] = {}

        # - scalar tensors -> metrics (floats)
        # - non-scalar tensors -> artifacts (lists)
        for k, v in out.items():
            py = tensor_to_python(v)
            if isinstance(py, float):
                metrics[k] = py
            else:
                artifacts[k] = py

        return MetricResult(metrics=metrics, artifacts=artifacts)

    def reset(self) -> None:
        self.metric.reset()
