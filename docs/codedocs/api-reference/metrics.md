---
title: "Metrics API"
description: "Reference for metric computers, metric orchestration, and prediction/target helpers."
---

The metrics package is defined in `src/raitap/metrics/` and exposes both low-level metric computers and a higher-level orchestration object that persists artifacts.

## Imports

```python
from raitap.metrics import (
    BaseMetricComputer,
    ClassificationMetrics,
    DetectionMetrics,
    MetricResult,
    Metrics,
    MetricsEvaluation,
    MetricsVisualizer,
    create_metric,
    evaluate,
    metrics_prediction_pair,
    metrics_run_enabled,
    resolve_metric_targets,
    scalar_metrics_for_tracking,
)
```

## Base types

### `MetricResult`

```python
@dataclass
class MetricResult:
    metrics: dict[str, float]
    artifacts: dict[str, Any] = field(default_factory=dict)
```

### `BaseMetricComputer`

```python
class BaseMetricComputer(ABC):
    def reset(self) -> None
    def update(self, predictions: Any, targets: Any) -> None
    def compute(self) -> MetricResult
```

## Concrete metric computers

### `ClassificationMetrics`

Source: `src/raitap/metrics/classification_metrics.py`

```python
def __init__(
    self,
    *,
    task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
    num_classes: int | None = None,
    num_labels: int | None = None,
    average: Literal["micro", "macro", "weighted", "none"] = "macro",
    ignore_index: int | None = None,
    **kwargs: Any,
) -> None
```

Public methods:

```python
def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None
def compute(self) -> MetricResult
def reset(self) -> None
```

### `DetectionMetrics`

Source: `src/raitap/metrics/detection_metrics.py`

```python
def __init__(
    self,
    *,
    box_format: Literal["xyxy", "xywh"] = "xyxy",
    iou_type: Literal["bbox", "segm"] | tuple[Literal["bbox", "segm"], ...] = "bbox",
    iou_thresholds: list[float] | None = None,
    rec_thresholds: list[float] | None = None,
    max_detection_thresholds: list[int] | None = None,
    class_metrics: bool = False,
    extended_summary: bool = False,
    average: Literal["macro", "micro"] = "macro",
    backend: Literal["pycocotools", "faster_coco_eval"] = "faster_coco_eval",
    **kwargs: Any,
) -> None
```

## Factory and orchestration

### `metrics_run_enabled`

```python
def metrics_run_enabled(config: AppConfig) -> bool
```

### `create_metric`

```python
def create_metric(metrics_config: Any) -> tuple[BaseMetricComputer, str]
```

### `Metrics`

```python
class Metrics:
    def __new__(cls, config: AppConfig, predictions: Any, targets: Any) -> MetricsEvaluation
```

Creates the configured metric computer, updates it once, writes metric artifacts, and returns a `MetricsEvaluation`.

### `MetricsEvaluation`

```python
@dataclass
class MetricsEvaluation(Trackable, Reportable):
    result: MetricResult
    run_dir: Path
    computer: BaseMetricComputer
    resolved_target: str
```

Public methods:

```python
def log(self, tracker: BaseTracker | None, *, prefix: str = "performance", **kwargs: Any) -> None
def to_report_group(self) -> ReportGroup
```

### `evaluate`

```python
def evaluate(config: AppConfig, predictions: Any, targets: Any) -> MetricsEvaluation
```

Thin wrapper around `Metrics(...)`.

## Prediction helpers

```python
def metrics_prediction_pair(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
def resolve_metric_targets(
    predictions: torch.Tensor,
    labels: torch.Tensor | None,
) -> torch.Tensor
def scalar_metrics_for_tracking(result: MetricResult) -> dict[str, float | int | bool]
```

Use these helpers when you want to feed the metrics package outside the full run pipeline.

## Visualisation helper

### `MetricsVisualizer`

```python
class MetricsVisualizer:
    @staticmethod
    def create_figures(result: MetricResult) -> dict[str, Figure]
```

Creates a metrics overview bar chart and an optional confusion matrix figure when the relevant artifact is present.
