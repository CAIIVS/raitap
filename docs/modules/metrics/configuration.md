```{config-page}
:intro: This page describes how to configure the metrics module used to score
  model predictions.

:option: _target_
:allowed: string
:default: "ClassificationMetrics"
:description: Hydra target for the metrics implementation.

:option: task
:allowed: string
:default: "multiclass"
:description: Task type passed to the metrics implementation.

:option: num_classes
:allowed: integer, null
:default: null
:description: Optional number of classes. Set this when it cannot be inferred
  safely.

:yaml:
metrics:
  _target_: "ClassificationMetrics"
  task: "multiclass"
  num_classes: 7

:cli: metrics.num_classes=7
```
