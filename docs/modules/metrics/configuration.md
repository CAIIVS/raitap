---
title: "Configuration"
description: "This page describes how to configure the metrics module used to score model predictions."
myst:
  html_meta:
    "description": "This page describes how to configure the metrics module used to score model predictions."
---

# Configuration

The `metrics` block scores model predictions. The `_target_` field is the
discriminator: it selects one of four adapters, each with its own set of valid
keys.

- **`BinaryClassificationMetrics`** — two-class problems (accuracy, precision,
  recall, F1).
- **`MulticlassClassificationMetrics`** — single-label multiclass
  classification (one correct class per sample).
- **`MultilabelClassificationMetrics`** — multilabel classification (multiple
  correct classes per sample, independent per-label decisions).
- **`DetectionMetrics`** — object detection (mean average precision and
  related summaries via torchmetrics).

The previous unified `ClassificationMetrics` target with a `task: binary | multiclass | multilabel`
field has been removed. Pick the task-specific adapter directly via `_target_`;
each adapter only accepts the keys documented for its section below.

See {doc}`frameworks-and-libraries` for the backend behaviour behind each
adapter.

## Binary classification

```{config-page}
:slug: binary
:intro: Configures `BinaryClassificationMetrics` for two-class problems.

:option: _target_
:allowed: "BinaryClassificationMetrics"
:default: "BinaryClassificationMetrics"
:description: Selects the binary-classification adapter.

:option: ignore_index
:allowed: integer, null
:default: null
:description: Optional target value to ignore when computing metrics.

:option: threshold
:allowed: float
:default: 0.5
:description: Decision threshold applied to predicted probabilities to obtain
  the positive class.

:yaml:
metrics:
  _target_: "BinaryClassificationMetrics"
  ignore_index: null
  threshold: 0.5

:cli: +metrics=binary_classification +metrics.threshold=0.6

:python:
from raitap.metrics import binary_classification

metrics = binary_classification(
    threshold=0.5,
)
```

## Multiclass classification

```{config-page}
:slug: multiclass
:intro: Configures `MulticlassClassificationMetrics` for single-label
  multiclass problems.

:option: _target_
:allowed: "MulticlassClassificationMetrics"
:default: "MulticlassClassificationMetrics"
:description: Selects the multiclass-classification adapter.

:option: num_classes
:allowed: integer
:default: required
:description: Number of classes. Must be positive. Required.

:option: average
:allowed: "micro", "macro", "weighted", "none"
:default: "macro"
:description: Aggregation mode passed to the underlying TorchMetrics
  implementations. See {doc}`frameworks-and-libraries` for semantics.

:option: ignore_index
:allowed: integer, null
:default: null
:description: Optional target value to ignore when computing metrics.

:yaml:
metrics:
  _target_: "MulticlassClassificationMetrics"
  num_classes: 7
  average: "macro"
  ignore_index: null

:cli: +metrics=multiclass_classification +metrics.num_classes=7

:python:
from raitap.metrics import multiclass_classification

metrics = multiclass_classification(
    num_classes=7,
    average="macro",
)
```

## Multilabel classification

```{config-page}
:slug: multilabel
:intro: Configures `MultilabelClassificationMetrics` for multilabel problems
  (independent per-label decisions).

:option: _target_
:allowed: "MultilabelClassificationMetrics"
:default: "MultilabelClassificationMetrics"
:description: Selects the multilabel-classification adapter.

:option: num_labels
:allowed: integer
:default: required
:description: Number of labels. Must be positive. Required.

:option: average
:allowed: "micro", "macro", "weighted", "none"
:default: "macro"
:description: Aggregation mode passed to the underlying TorchMetrics
  implementations.

:option: ignore_index
:allowed: integer, null
:default: null
:description: Optional target value to ignore when computing metrics.

:option: threshold
:allowed: float
:default: 0.5
:description: Per-label decision threshold applied to predicted probabilities.

:yaml:
metrics:
  _target_: "MultilabelClassificationMetrics"
  num_labels: 5
  average: "macro"
  ignore_index: null
  threshold: 0.5

:cli: +metrics=multilabel_classification +metrics.num_labels=5

:python:
from raitap.metrics import multilabel_classification

metrics = multilabel_classification(
    num_labels=5,
    average="macro",
    threshold=0.5,
)
```

## Object detection

```{config-page}
:slug: detection
:intro: Configures `DetectionMetrics`, which wraps torchmetrics
  `MeanAveragePrecision`. The IoU-related knobs are grouped under the nested
  `iou:` block.

:option: _target_
:allowed: "DetectionMetrics"
:default: "DetectionMetrics"
:description: Selects the object-detection adapter.

:option: box_format
:allowed: "xyxy", "xywh"
:default: "xyxy"
:description: Bounding-box format of incoming predictions and targets.
  torchvision detectors output `xyxy`.

:option: iou.type
:allowed: "bbox", "segm", or a tuple of those values
:default: "bbox"
:description: IoU mode passed to mean average precision.

:option: iou.thresholds
:allowed: list[float], null
:default: null
:description: Optional custom IoU thresholds. `null` uses the torchmetrics
  default (COCO-style sweep).

:option: iou.rec_thresholds
:allowed: list[float], null
:default: null
:description: Optional custom recall thresholds. `null` uses the torchmetrics
  default.

:option: iou.max_detection_thresholds
:allowed: list[int], null
:default: null
:description: Optional maximum-detection cutoffs. `null` uses the torchmetrics
  default.

:option: class_metrics
:allowed: true, false
:default: false
:description: Whether to compute per-class metrics in addition to global
  summaries.

:option: extended_summary
:allowed: true, false
:default: false
:description: Whether to request the torchmetrics extended (non-scalar)
  summary outputs as artifacts.

:option: average
:allowed: "macro", "micro"
:default: "macro"
:description: Aggregation mode for detection summaries.

:option: backend
:allowed: "faster_coco_eval", "pycocotools"
:default: "faster_coco_eval"
:description: Backend used by mean average precision. `pycocotools` should be
  used only for backward compatibility.

:yaml:
metrics:
  _target_: "DetectionMetrics"
  box_format: "xyxy"
  iou:
    type: "bbox"
    thresholds: null
    rec_thresholds: null
    max_detection_thresholds: null
  class_metrics: false
  extended_summary: false
  average: "macro"
  backend: "faster_coco_eval"

:cli: +metrics=detection +metrics.class_metrics=true +metrics.extended_summary=true

:python:
from raitap.metrics import detection

metrics = detection(
    box_format="xyxy",
    iou={"type": "bbox"},
    class_metrics=True,
    extended_summary=True,
)
```
