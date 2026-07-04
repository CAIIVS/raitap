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

:cli: data.source="./data/images" data.preprocessing=model-bundled data.model_input_transformation=model-bundled

:python:
from raitap.data import DataConfig

data = DataConfig(
    name="my-dataset",
    description="Internal validation set",
    source="./data/images",
    forward_batch_size=32,
    preprocessing="model-bundled",
    model_input_transformation="model-bundled",
)
```

**Label variants.** `data.labels` is a Hydra config-group: select the variant with `defaults: [data/labels: <name>]`, then set its fields under `data.labels:`. Each variant exposes only the fields it accepts — setting a foreign field is a load error. Prefer the `defaults` group: it validates fields at config-load with a clear error. Inlining `_target_` directly (e.g. `data.labels: {_target_: TabularLabelParser, bogus: 1}`) skips that load-time struct check, so a foreign field is not caught until the parser is built and then fails with a less obvious instantiation error.

```yaml
defaults:
  - raitap_schema
  - data/labels: tabular   # pick one variant
  - _self_

data:
  source: "./data/images"
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"
```

| Variant          | Task(s)                    | Fields                                                  |
| ---------------- | -------------------------- | ------------------------------------------------------- |
| `tabular`        | classification             | `source`, `id_column`, `column`, `encoding`, `id_strategy` |
| `directory`      | classification             | *(none — class from top-level subdir name)*             |
| `coco`           | detection + classification | `source`, `id_strategy`                                 |
| `yolo`           | detection                  | `source`, `id_strategy`                                 |
| `voc`            | detection                  | `source`, `id_strategy`, `class_names`                  |
| `detection_json` | detection                  | `source`, `id_strategy`                                 |

**`tabular`** — CSV, TSV, or Parquet file. `id_column` aligns rows to sample filenames; `column` names the label column (omit for one-hot numeric columns, which are reduced with `argmax`). `encoding` is one of `"index"`, `"one_hot"`, `"argmax"`. `id_strategy` controls alignment — see below.

**`directory`** — no labels file. Class is each sample's top-level subdirectory (torchvision `ImageFolder` style). See {doc}`own-vs-built-in`.

**`coco`** — single `instances.json` file (`source`). Category ids pass through unchanged. Serves detection and classification.

**`yolo`** — directory of per-image `.txt` files (`source`). Needs `data.source` set to the image directory so RAITAP can match annotation files to images. Detection only. Category ids pass through unchanged.

**`voc`** — directory of per-image `.xml` files (`source`). `class_names` maps VOC names to integer ids; falls back to `model.class_names`, then the standard 20-class VOC order. Detection only.

**`detection_json`** — RAITAP native JSON record list `[{"sample_id": ..., "boxes": ..., "labels": ...}]`. Detection only.

All detection variants honour `id_strategy` for nested image-directory layouts.

**`id_strategy`** (`"auto"` / `"relative_path"` / `"stem"`, default `"auto"`): how label ids are matched against discovered sample files. `"auto"` inspects the id column and switches to `"relative_path"` if any value contains `/` or `\`, otherwise `"stem"`. `"relative_path"` keeps directory components (e.g. `NORMAL/IM-0001`) — required when filename stems collide across class subdirs. `"stem"` matches by basename only.

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
