from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from raitap.configs.schema import DetectionMetricsConfig, IoUConfig
from raitap.metrics.registration import metrics_adapter
from raitap.utils.lazy import lazy_import

from .base_metric_computer import BaseMetricComputer, MetricResult
from .utils import tensor_to_python

if TYPE_CHECKING:
    import torch
    from torchmetrics.detection import mean_ap as _tm_detection_mean_ap
else:
    # Deferred so partial-extras venvs (no ``metrics`` extra) can still import
    # this module — needed by ``raitap.run(..., auto_install_deps=True)`` and the AST adapter scan.
    # Bind the proxy itself (not ``.MeanAveragePrecision``) so attribute access
    # happens at instantiation time inside ``__init__``, not at module load.
    _tm_detection_mean_ap = lazy_import("torchmetrics.detection.mean_ap")

BoxFormat = Literal["xyxy", "xywh"]  # torchvision outputs xyxy
IoUType = Literal["bbox", "segm"] | tuple[Literal["bbox", "segm"], ...]
Backend = Literal["pycocotools", "faster_coco_eval"]
Average = Literal["macro", "micro"]


def _coerce_iou(iou: IoUConfig | dict[str, Any] | None) -> IoUConfig:
    if iou is None:
        return IoUConfig()
    if isinstance(iou, IoUConfig):
        return iou
    return IoUConfig(**iou)


@metrics_adapter(
    registry_name="detection",
    extra="metrics",
    schema=DetectionMetricsConfig,
)
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
        iou: IoUConfig | dict[str, Any] | None = None,
        class_metrics: bool = False,
        extended_summary: bool = False,
        average: Average = "macro",
        backend: Backend = "faster_coco_eval",
        **kwargs: Any,
    ):
        iou_cfg = _coerce_iou(iou)
        self.metric = _tm_detection_mean_ap.MeanAveragePrecision(
            box_format=box_format,
            iou_type=iou_cfg.type,
            iou_thresholds=iou_cfg.thresholds,
            rec_thresholds=iou_cfg.rec_thresholds,
            max_detection_thresholds=iou_cfg.max_detection_thresholds,
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

        return MetricResult(scalars=metrics, artifacts=artifacts)

    def reset(self) -> None:
        self.metric.reset()
