# Underlying libraries

The current metrics module relies on:

- `torchmetrics` for classification and detection metric implementations
- `faster-coco-eval` as the default backend for detection mean average
  precision

To tweak specific options via the RAITAP config, you might need to refer to the underlying library's documentation.

## Classification metrics

`ClassificationMetrics` is ideal for classification models. It wraps the following TorchMetrics classes:

- [`Accuracy`](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html)
- [`Precision`](https://lightning.ai/docs/torchmetrics/stable/classification/precision.html)
- [`Recall`](https://lightning.ai/docs/torchmetrics/stable/classification/recall.html)
- [`F1Score`](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html)

It supports the task types exposed by TorchMetrics that RAITAP currently uses:

- [`binary`](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#binaryaccuracy)
- [`multiclass`](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#multiclassaccuracy)
- [`multilabel`](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#multilabelaccuracy)

RAITAP adds a thin layer of validation and conventions around these metrics. In
particular:

- `multiclass` requires `num_classes`
- `multilabel` uses `num_labels` internally and also accepts `num_classes` as an alias. It defaults to a threshold of `0.5` if none is provided
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
