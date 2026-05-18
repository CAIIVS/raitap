---
title: "Contributing to the data module"
description: "This page describes how to add new built-in sample datasets to RAITAP."
myst:
  html_meta:
    "description": "This page describes how to add new built-in sample datasets to RAITAP."
---

# Contributing to the data module

This page describes how to add new built-in sample datasets to RAITAP.

## Overview

RAITAP ships named sample bundles (e.g. `imagenet_samples`, `mnist_samples`)
referenceable by name in `data.source`. Registration lives in
`src/raitap/data/samples.py` as plain dicts:

- `SAMPLE_SOURCES` — name → list of `(url, filename)` pairs, downloaded into
  the per-user cache (`~/.cache/raitap/<name>/`).
- `SAMPLE_LABELS` (optional) — name → `{filename: class_index}` map,
  materialised as `labels.csv` alongside the images so
  `data.labels.source: <name>` resolves automatically.

## Adding a built-in sample dataset

1. **Register the source files** in `src/raitap/data/samples.py`:

    ```python
    SAMPLE_SOURCES["cifar10_samples"] = [
        ("https://…/airplane.png", "airplane.png"),
        ("https://…/cat.png", "cat.png"),
        # …
    ]
    ```

2. **(Optional) Register labels** if the dataset has a known ground truth
   matching the model's class space:

    ```python
    SAMPLE_LABELS["cifar10_samples"] = {
        "airplane.png": 0,
        "cat.png": 3,
        # …
    }
    ```

3. **Use it** from any consumer config:

    ```yaml
    data:
      name: cifar10_samples
      source: cifar10_samples            # resolves via SAMPLE_SOURCES
      labels:
        source: cifar10_samples          # resolves via SAMPLE_LABELS → labels.csv
        id_column: image
        column: label
    ```

    Or override on the CLI:

    ```bash
    uv run raitap --config-name assessment data.source=cifar10_samples
    ```

4. **Add custom loading logic** in `src/raitap/data/data.py` or `samples.py`
   only if the bundle needs more than the default image/tabular loader.

5. **Update docs** — add the new sample name to
   {doc}`/modules/data/own-vs-built-in`.

## Sample discovery and label alignment

`data.source` directories are walked **recursively** (`Path.rglob`); sample
ids are posix-style paths relative to the source root (e.g.
`NORMAL/IM-0001.jpeg`). Supports nested `ImageFolder`-style layouts where
filename stems collide across class subdirs.

Label alignment is governed by `data.labels.id_strategy`:

- `"auto"` (default) — inspects the id column once. If any value contains
  `/` or `\`, switches to `"relative_path"`; otherwise `"stem"`.
- `"relative_path"` — strips only the file extension; directory components
  retained.
- `"stem"` — legacy flat-dir behaviour (`Path(id).stem`).

Both sides (sample ids from disk + label ids from the file) are normalised
the same way before lookup. Duplicate normalised label ids raise a
warning and disable label-based metrics for that run.

## Image preprocessing internals

User-facing surface lives at {doc}`/modules/data/preprocessing` — that doc is
the source of truth for vocabulary exposed to users (no class names, no
internal mechanics). This section covers the implementation so maintainers can
extend it without re-deriving the design.

### Two independent knobs

`DataConfig` exposes two independent knobs, each accepting `None`,
`"model-bundled"`, or a path to a `.py` file:

| Knob                              | Side  | Typical contents      |
| --------------------------------- | ----- | --------------------- |
| `data.preprocessing`              | data  | Resize, CenterCrop    |
| `data.model_input_transformation` | model | Normalize             |

Per-knob origin (`off` / `model-bundled` / `custom-file`) is recorded on
`ResolvedPreprocessing.data_origin` and `.model_origin`.

### Split: data preprocessing vs model input transformation

The resolved preprocessing has two modules, one for the data loader and one
for the model boundary:

| Half           | Where                        | What                        | Why                                                                                                                                  |
| -------------- | ---------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `data_module`  | `Data._load_data`, per image | Data preprocessing          | Must run before `_stack_images_numpy` calls `np.stack` so mixed-size directories can be batched. No gradient → safe outside autograd. |
| `model_module` | `Model._apply_preprocessing` | Model input transformation | Has gradient → must stay inside autograd for Captum / SHAP / torchattacks to attribute and attack on the user-facing `[0, 1]` space. |

For ONNX backends, `model_module` still belongs at the model boundary, but
the autograd rationale is Torch-specific. Marabou-style formal verification
reads the bare ONNX graph and does not see Python preprocessing modules.

`model-bundled` on either knob calls `_split_preset(preset)` for the bundled
torchvision presets (`ImageClassification`, `SemanticSegmentation`) and
selects the relevant half — Resize+CenterCrop for the data side, Normalize
for the model side. The preset is computed once per arch per
`resolve_preprocessing` call (cached internally). `ObjectDetection` returns
both halves as `None` — detection models normalise internally via
`GeneralizedRCNNTransform`.

`custom-file` on a knob imports the file (cached when the same path is used
on both knobs, so one disk read + hash) and looks for the matching
decorator: `@raitap_preprocessing_factory` for the data knob,
`@raitap_model_input_transformation_factory` for the model knob. Missing
the matching decorator raises `AttributeError`; multiple matching
factories raise `ValueError`. A file with both decorators referenced by
only one knob silently ignores the other (file reuse is the point).

`off` on a knob produces `module=None` for that side. If both knobs are
`off` and inputs are images, `_OFF_WARNING` fires. If only the model knob
is `off` with images, `_MODEL_OFF_WARNING` fires (the silent
metric-corruption case).

### Resolver flow

Entry point: `resolve_preprocessing(model_cfg, data_cfg) -> ResolvedPreprocessing`
in `src/raitap/data/preprocessing.py`. The resolver is side-effect-free — it
inspects the config, returns a `ResolvedPreprocessing` record, and leaves
panel rendering / warning emission to its caller.

```
DataConfig.preprocessing                  (data side)
  ├─ None / ""           → _SideResolved(module=None, origin="off", ...)
  ├─ "model-bundled"     → _resolve_side_bundled → split_preset()[data half]
  └─ <path>.py           → _resolve_side_custom_file → @raitap_preprocessing_factory

DataConfig.model_input_transformation     (model side)
  ├─ None / ""           → _SideResolved(module=None, origin="off", ...)
  ├─ "model-bundled"     → _resolve_side_bundled → split_preset()[model half]
  └─ <path>.py           → _resolve_side_custom_file → @raitap_model_input_transformation_factory
```

The two sides are composed into a single `ResolvedPreprocessing`. Off
warnings (both off, or model-only off with images) fire from the composer,
not from per-side resolution.

The normal pipeline resolves preprocessing once in
`src/raitap/pipeline/orchestrator.py`, then passes the same
`ResolvedPreprocessing` to `Model` and `Data`. Direct `Model(config)` and
`Data(config)` callers still fall back to resolving internally.

`model-bundled` derives the torchvision arch from `model_cfg.arch`,
falling back to `model_cfg.source` if it names a built-in
(`hasattr(torchvision.models, name)`). Missing-arch lineage raises
`ValueError` with a copy-pasteable fix. This path is intentionally
torchvision-lineage only; RAITAP does not inspect or execute preprocessing
that may have been bundled into an ONNX export. `custom-file` enforces
`.py` extension, existence, decorated factory discovery, zero-arg factory
signatures, and `nn.Module` return values; each failure raises a typed error.

### Wrap insertion

`Model.__init__` calls `_apply_preprocessing(self.backend, config)` in
`src/raitap/models/model.py`. When `resolved.model_module is not None` and
the backend is a `TorchBackend`, the helper moves the model input
transformation to the backend's device, switches it to `eval()`, and
rebinds:

```python
backend.model = nn.Sequential(model_module, backend.model)
```

`nn.Sequential` keeps the wrap invisible to autograd, captum, shap, and
torchattacks — gradients flow back through the normalize layer, attack
budgets stay defined in the user-facing `[0, 1]` input space.

When the backend is an `OnnxBackend`, `_apply_preprocessing` accepts only
a `custom-file` model-side resolution (`model_origin == "custom-file"`). It
deep-copies the resolved `model_module` and passes it to
`backend.set_preprocessing(model_module)`, whose setter owns CPU placement
and `eval()` mode. `model_input_transformation: model-bundled` stays
unsupported for ONNX because there is no torchvision weights lineage to
derive from.

After wrapping, the helper emits each `resolved.warnings` entry via
`raitap_log.warn` and (for active tiers) a single `raitap_log.info` line
with `resolved.description`. Panel rendering is handled by the standard
log handler.

### Loader integration

`Data._load_data` (`src/raitap/data/data.py`) receives the run-level
`ResolvedPreprocessing` when called by the orchestrator, lifts
`resolved.data_module` to a per-image callable via
`module_as_per_image_callable` (which runs in `torch.no_grad` + `eval()`),
and passes it into `_stack_images_numpy(files, per_image_transform=...)`.
The transform runs on each `(C, H, W)` tensor before `np.stack`, so a
directory of mixed-size JPEGs becomes a clean `(N, 3, 224, 224)` batch.

Sample-source paths (`SAMPLE_SOURCES`) get data preprocessing applied to the
already-stacked tensor returned by `_load_sample` — torchvision presets
accept batched tensors so this works identically. Defensive: if `cfg.model`
is absent (minimal test mocks), preprocessing resolution is skipped entirely
and `per_image_transform = None`.

### ONNX path

ONNX support is split by `model_origin`:

- `off`: unchanged; the resolver still runs so the off warning can be
  emitted for ONNX-with-no-preprocessing setups.
- `model-bundled`: unsupported. The bundled split is torchvision-lineage
  only and does not apply preprocessing that may have been bundled into an
  ONNX export.
- `custom-file`: supported on RAITAP's normal tensor/model call path. The
  decorated model input transformation runs before the backend invocation,
  so ONNX models get the same model-boundary transformation semantics as
  Torch models.

The low-level `OnnxBackend.forward_numpy(...)` API remains raw by design: it
executes the ONNX session on the array it is given and does not resolve or
apply Python preprocessing. Callers that bypass `Model` / the orchestrator
must apply preprocessing themselves.

### Consent surfaces

The `custom-file` option executes arbitrary Python from a user-named file.
Two consent mechanisms exist, and `_consent_given()` accepts either:

| Path         | Mechanism                                                                                                              |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| CLI          | `--allow-preprocessing-exec` / `-yp` parsed by `_strip_deps_flags` in `src/raitap/deps/bootstrap.py`; on hit, bootstrap sets `os.environ["RAITAP_ALLOW_PREPROCESSING_EXEC"] = "1"` before Hydra |
| Programmatic | `acknowledge_preprocessing_exec: bool = False` kwarg on `raitap.run` (`src/raitap/api.py`), threaded through the orchestrator to `resolve_preprocessing(..., acknowledge_exec=...)`            |

Both are required because `raitap.run(config)` (`src/raitap/api.py`) bypasses
the bootstrap layer entirely — a CLI-only gate would silently allow exec on
the programmatic path. The off-warning suppressor mirrors this shape:
`--acknowledge-preprocessing-off` on CLI (env var
`RAITAP_ACKNOWLEDGE_PREPROCESSING_OFF`), `acknowledge_preprocessing_off=True`
kwarg on `raitap.run` programmatically.

The env vars are also re-exported through `_exec` so dep-bootstrap re-execs
preserve consent across the sub-process boundary.

### `ResolvedPreprocessing` shape

```python
@dataclass(frozen=True)
class ResolvedPreprocessing:
    data_module: nn.Module | None              # Data preprocessing — per-image in loader
    model_module: nn.Module | None             # Model input transformation — model boundary
    data_origin: Literal["off", "model-bundled", "custom-file"]
    model_origin: Literal["off", "model-bundled", "custom-file"]
    description: str                           # composite "data: …; model: …"
    data_file_path: Path | None = None         # data side custom-file only
    data_file_sha256: str | None = None        # data side custom-file only
    model_file_path: Path | None = None        # model side custom-file only
    model_file_sha256: str | None = None       # model side custom-file only
    warnings: list[str] = field(default_factory=list)
```

`is_active` is `True` if either module is non-None.

Downstream consumers:

| Consumer                                           | Reads                                          |
| -------------------------------------------------- | ---------------------------------------------- |
| `Data._load_data`                                  | `data_module` (lifted via `module_as_per_image_callable`) |
| `_apply_preprocessing` wrap + panel + warnings     | `model_module`, `model_origin`, `description`, `warnings` |
| `_mlflow_summary_params` in `mlflow_tracker.py`    | `data.preprocessing` and `data.model_input_transformation`, each emitting `.origin` / `.file_path` / `.file_sha256` keys |
| Future HTML report card                            | All fields — same plain phrasing as the panel  |

### Test fixture

`src/raitap/data/tests/fixtures/preproc_imagenet.py` is the minimal
custom-file model input transformation exercised by `test_preprocessing.py`
and `test_api.py`.
It mirrors the example in {doc}`/modules/data/preprocessing` so the test
doubles as documentation that the example works — keep them in sync when
either changes.
