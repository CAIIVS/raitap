---
title: "Configuration"
description: "For tabular models whose backend expects an unusual per-sample layout (such as ACAS Xu, a Torch network whose forward takes (N, 1, 1, 5)), supply input_metadata.shape explicitly so the pipeline reshapes the flat feature vectors before the…"
myst:
  html_meta:
    "description": "For tabular models whose backend expects an unusual per-sample layout (such as ACAS Xu, a Torch network whose forward takes (N, 1, 1, 5)), supply inputmetadata.shape explicitly so the pipeline reshapes the flat feature vectors before the fo"
---

```{config-page}
:intro: This page describes how to configure the data used to assess the model.

:option: name
:allowed: string
:default: "imagenet_samples"
:description: Name shown in outputs and tracking metadata. 

:option: description
:allowed: string, null
:default: null
:description: Optional human-readable dataset description.

:option: source
:allowed: string, null
:default: null
:description: Path to a local data directory or file, a URL, or a named
  sample set such as `"imagenet_samples"`. Sample names download on first
  use into `~/.cache/raitap/<name>/`; the same name is also accepted by
  `labels.source` when the sample bundles ground-truth labels.

:option: forward_batch_size
:allowed: int, null
:default: null
:description: Optional batch size for the initial model forward pass used to
  produce predictions for metrics, report sample ranking, and `auto_pred`
  explainer targets. If omitted, RAITAP uses its pipeline default of `32`.
  This does not control explainer attribution batching; use
  `transparency.<explainer>.raitap.batch_size` for that.

:option: preprocessing
:allowed: null, "model-bundled", path string
:default: null
:description: Data-side preprocessing applied in the loader, before the
  batch reaches the model. Per-image for image sources; per-batch on the
  stacked `(N, F)` tensor for tabular sources. Typical contents: Resize +
  CenterCrop for images; feature scaling or encoding for tabular. `null`
  leaves the loader untouched; `"model-bundled"` pulls Resize + CenterCrop
  from the model's bundled torchvision preset (image models only); a `.py`
  path loads a factory decorated with `@raitap_preprocessing_factory`. The
  `.py` path requires consent — see
  <a href="../../using-raitap/flags.html#flag-allow-preprocessing-exec"><code>--allow-preprocessing-exec</code></a>
  and {doc}`preprocessing`.

:option: model_input_transformation
:allowed: null, "model-bundled", path string
:default: null
:description: Transformation applied at the model boundary, on every
  forward pass. Typical contents: Normalize. For Torch backends it stays
  inside autograd so attribution and adversarial budgets see the
  user-facing input space; for ONNX backends it runs on the tensor call
  path before the ONNX session. `"model-bundled"` pulls Normalize from the
  model's bundled torchvision preset (Torch backends only — ONNX has no
  torchvision lineage to derive from); a `.py` path loads a factory
  decorated with `@raitap_model_input_transformation_factory` and works
  for both Torch and ONNX backends. When both this and `preprocessing` are
  `null` and inputs are images, a loud warning fires — silence it with
  <a href="../../using-raitap/flags.html#flag-acknowledge-preprocessing-off"><code>--acknowledge-preprocessing-off</code></a>.
  See {doc}`preprocessing`.

:option: labels.source
:allowed: string, null
:default: null
:description: Optional path to a labels file (CSV, TSV, or Parquet), URL, or
  named sample set. When set to a sample name (e.g. `"imagenet_samples"`),
  raitap resolves to the labels CSV bundled with that sample. Sample sets
  without bundled labels raise an error. The reserved value `"directory"`
  derives classification labels from each sample's top-level class
  subdirectory (torchvision `ImageFolder` style; no labels file) — see
  {doc}`own-vs-built-in`. In that mode `id_column` and `id_strategy` do not
  apply.

:option: labels.id_column
:allowed: string, null
:default: null
:description: Optional sample-ID column used to align labels with filenames,
  for example `"image"`.

:option: labels.column
:allowed: string, null
:default: null
:description: Optional class-label column. If omitted, one-hot numeric columns
  are reduced with `argmax`.

:option: labels.encoding
:allowed: "index", "one_hot", "argmax", null
:default: null
:description: Optional label parsing strategy.

:option: labels.id_strategy
:allowed: "auto", "relative_path", "stem"
:default: "auto"
:description: How label-file ids are matched against discovered sample
  files. `"auto"` (default) inspects the id column and switches to
  `"relative_path"` if any value contains `/` or `\`, otherwise falls back
  to `"stem"`. `"relative_path"` keeps directory components and supports
  nested ImageFolder layouts (e.g. `NORMAL/IM-0001.jpeg`) — required when
  filename stems collide across class subdirs. `"stem"` matches by basename only (flat-dir layouts).

:option: labels.format
:allowed: "native", "coco", "yolo", "voc"
:default: "native"
:description: External label file format. `"native"` (default) reads RAITAP's
  own shape (classification: CSV/TSV/Parquet or the `"directory"` source;
  detection: the JSON record list). `"coco"`, `"yolo"`, and `"voc"` convert a
  standard annotation file to the native shape before alignment. `"yolo"` and
  `"voc"` are detection only; `"coco"` serves detection and classification.
  Non-native formats align by sample id, so a labels id is required.

:option: input_metadata
:allowed: dict, null
:default: null
:description: Input modality + layout hints, normally auto-inferred from
  `data.source` for image and tabular sources. Set this explicitly when the
  auto-inference cannot resolve the layout (e.g. raw tensors, custom loaders,
  or ONNX/Torch models whose declared input shape differs from the on-disk
  sample shape). Keys: `kind` (one of `image`, `tabular`, `text`,
  `time_series`), `layout` (`NCHW`, `(B,F)`, `(B,T,C)`, `TOKENS`),
  `feature_names`, and `shape`.

:option: input_metadata.shape
:allowed: list[int], null
:default: null
:description: The **non-batch** per-sample layout expected by the model
  (batch dim is implicit and resolved at runtime). Beyond informing
  output-space inference, `shape` also controls the model-input reshape
  performed at the backend boundary: per-sample inputs whose `numel` matches
  `shape` are reshaped to `(N, *shape)` before being passed to the model.
  For ONNX models the backend auto-derives the expected shape from
  `session.get_inputs()[0].shape` — concrete dims are respected as-is and
  symbolic / unknown dims (e.g. `"batch"`) become dynamic.
  `input_metadata.shape` overrides the auto-derived value, useful when an
  ONNX graph declares the batch dim as a fixed `1` even though the model
  accepts arbitrary batches, or when you want to feed flatter inputs than
  the graph declares. For Torch models, `nn.Module` declares no shape, so
  setting `input_metadata.shape` is the only way to enable backend reshape —
  otherwise inputs are passed through unchanged. A
  `raitap.utils.errors.ModelInputShapeError` is raised when per-sample
  `numel` mismatches the expected shape, or when an ONNX graph declares two
  or more dynamic dims (ambiguous — supply `shape` to disambiguate).

:yaml:
data:
  name: "my-dataset"
  description: "Internal validation set"
  source: "./data/images"
  forward_batch_size: 32
  preprocessing: model-bundled
  model_input_transformation: model-bundled
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"
    encoding: "index"
    id_strategy: "auto"

:cli: data.source="./data/images" data.preprocessing=model-bundled data.model_input_transformation=model-bundled data.labels.source="./data/labels.csv" data.labels.column=label

:python:
from raitap.data import (
    DataConfig,
    IdStrategy,
    LabelEncoding,
    LabelsConfig,
    Preprocessing,
)

data = DataConfig(
    name="my-dataset",
    description="Internal validation set",
    source="./data/images",
    forward_batch_size=32,
    preprocessing=Preprocessing.model_bundled,
    model_input_transformation=Preprocessing.model_bundled,
    labels=LabelsConfig(
        source="./data/labels.csv",
        id_column="image",
        column="label",
        encoding=LabelEncoding.index,
        id_strategy=IdStrategy.auto,
    ),
)
```

## Label formats

RAITAP reads common annotation formats directly via `data.labels.format`.

| Format   | Detection | Classification | Source layout                                  |
| -------- | --------- | -------------- | ---------------------------------------------- |
| `native` | yes       | yes            | JSON record list / CSV-TSV-Parquet             |
| `coco`   | yes       | yes            | single `instances.json`                        |
| `yolo`   | yes       | no             | dir of per-image `.txt` (needs `data.source`)  |
| `voc`    | yes       | no             | dir of per-image `.xml`                        |

COCO and YOLO labels keep their category ids unchanged. VOC class names map to
ids by `model.class_names` order, else the standard 20-class VOC order.

Detection formats match each record's `sample_id` against the discovered image
file by exact name, so the image directory must be flat (nested subdirs are not
matched). Classification labels still align via `labels.id_strategy`.

For tabular models whose backend expects an unusual per-sample layout (such
as ACAS Xu, a Torch network whose forward takes `(N, 1, 1, 5)`), supply
`input_metadata.shape` explicitly so the pipeline reshapes the flat feature
vectors before the forward pass:

```yaml
data:
  input_metadata:
    kind: tabular
    layout: "(B,F)"
    feature_names: [rho, theta, psi, v_own, v_int]
    shape: [1, 1, 5]   # non-batch dims; reshapes (N, 5) -> (N, 1, 1, 5)
```

For nested `ImageFolder`-style layouts (e.g. `data/test/<class>/<file>.jpg`)
see {doc}`own-vs-built-in`.
