---
title: "Underlying libraries"
description: "The current metrics module relies on:"
myst:
  html_meta:
    "description": "The current metrics module relies on:"
---

# Underlying libraries

The current metrics module relies on:

- `torchmetrics` for classification and detection metric implementations
- `faster-coco-eval` as the default backend for detection mean average
  precision

To tweak specific options via the RAITAP config, you might need to refer to the underlying library's documentation.

## Adapter reference

Every `use:` key registered for the metrics module, generated from the
adapter registry.

```{include} ../_generated_adapters.md
:start-after: <!-- raitap-adapters:metrics:start -->
:end-before: <!-- raitap-adapters:metrics:end -->
```

## Classification metrics

`BinaryClassificationMetrics`, `MulticlassClassificationMetrics`, and `MultilabelClassificationMetrics` are the task-specific adapters for classification models. They each wrap the following TorchMetrics classes (instantiated for the matching task):

- [`Accuracy`](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html)
- [`Precision`](https://lightning.ai/docs/torchmetrics/stable/classification/precision.html)
- [`Recall`](https://lightning.ai/docs/torchmetrics/stable/classification/recall.html)
- [`F1Score`](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html)

The adapters map to the task types exposed by TorchMetrics that RAITAP currently uses:

- `BinaryClassificationMetrics` → [`binary`](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#binaryaccuracy)
- `MulticlassClassificationMetrics` → [`multiclass`](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#multiclassaccuracy)
- `MultilabelClassificationMetrics` → [`multilabel`](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#multilabelaccuracy)

RAITAP adds a thin layer of validation and conventions around these metrics. In
particular:

- `MulticlassClassificationMetrics` requires `num_classes`
- `MultilabelClassificationMetrics` requires `num_labels` and defaults to a threshold of `0.5` if none is provided
- `average="none"` stores per-class or per-label values in `artifacts.json`
  instead of flattening them into scalar metrics

For typical classification runs, the scalar results written to
`metrics/metrics.json` are `accuracy`, `precision`, `recall`, and `f1`.

## Detection metrics

`DetectionMetrics` is ideal for object detection models. It wraps TorchMetrics'
[`MeanAveragePrecision`](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html).

Detection inputs follow the structure expected by TorchMetrics: a list of
prediction dictionaries and a list of target dictionaries, usually one entry
per image. Predictions include `boxes`, `scores`, and `labels`, while targets
include `boxes` and `labels`.

Scalar detection results such as `map` are written to `metrics.json`. Larger
structured outputs, such as class-wise or extended summaries, are written to
`artifacts.json`.

## Third-party adapters

Third-party adapters published to PyPI can register under the `raitap.adapters`
entry-point group and are auto-discovered at config-registration time. Once
installed they appear alongside in-tree metric computers. See
{doc}`../../contributor/writing-a-plugin`.
