```{config-page}
:intro: This page describes how to configure the metrics module used to score
  model predictions.

:option: _target_
:allowed: "ClassificationMetrics", "DetectionMetrics"
:default: "ClassificationMetrics"
:description: The type of metrics to use. See [Frameworks and libraries](frameworks-and-libraries.md) for more details.

:option: task
:allowed: "binary", "multiclass", "multilabel"
:default: "multiclass"
:description: Classification task type. Only used by
  `ClassificationMetrics`.

:option: num_classes
:allowed: integer, null
:default: null
:description: Number of classes for `multiclass` tasks. For
  `multilabel`, this is also accepted as an alias for `num_labels`.

:option: num_labels
:allowed: integer, null
:default: null
:description: Number of labels for `multilabel` tasks.

:option: average
:allowed: "micro", "macro", "weighted", "none"
:default: "macro"
:description: Aggregation mode. `ClassificationMetrics` supports
  `"micro"`, `"macro"`, `"weighted"`, and `"none"` for `multiclass` and
  `multilabel`. `DetectionMetrics` supports `"micro"` and `"macro"`. See [Frameworks and libraries](frameworks-and-libraries.md) for more details.

:option: ignore_index
:allowed: integer, null
:default: null
:description: Optional target value to ignore when computing classification
  metrics.

:option: threshold
:allowed: float, null
:default: 0.5 for `multilabel` when omitted
:description: Threshold applied by `ClassificationMetrics` for multilabel
  predictions. Ignored by other targets unless the underlying TorchMetrics
  implementation accepts it through forwarded kwargs.

:option: box_format
:allowed: "xyxy", "xywh"
:default: "xyxy"
:description: Bounding-box format expected by `DetectionMetrics`.

:option: iou_type
:allowed: "bbox", "segm", or a tuple containing those values
:default: "bbox"
:description: IoU mode passed to detection mean average precision.

:option: iou_thresholds
:allowed: list[float], null
:default: null
:description: Optional custom IoU thresholds for detection evaluation.

:option: rec_thresholds
:allowed: list[float], null
:default: null
:description: Optional custom recall thresholds for detection evaluation.

:option: max_detection_thresholds
:allowed: list[int], null
:default: null
:description: Optional maximum-detection cutoffs for detection evaluation.

:option: class_metrics
:allowed: true, false
:default: false
:description: Whether `DetectionMetrics` should compute class-wise metrics in
  addition to global summaries.

:option: extended_summary
:allowed: true, false
:default: false
:description: Whether `DetectionMetrics` should request extended non-scalar
  summary outputs.

:option: backend
:allowed: "faster_coco_eval", "pycocotools"
:default: "faster_coco_eval"
:description: Backend used by detection mean average precision. `pycocotools` should be used only for backward compatibility.

:yaml:
metrics:
  _target_: "ClassificationMetrics"
  task: "multiclass"
  num_classes: 7
  num_labels: null
  average: "macro"
  ignore_index: null

:cli: metrics=detection metrics.class_metrics=true metrics.extended_summary=true
```
