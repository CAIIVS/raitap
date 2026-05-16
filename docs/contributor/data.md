# Contributing to the data module

This page describes how to add new built-in sample datasets to RAITAP.

## Overview

RAITAP ships a small set of named sample bundles (e.g. `imagenet_samples`,
`mnist_samples`) that can be referenced by name in `data.source`. The registration lives in `src/raitap/data/samples.py` as plain
Python dicts:

- `SAMPLE_SOURCES` — name → list of `(url, filename)` pairs to download into
  the per-user cache (`~/.cache/raitap/<name>/`).
- `SAMPLE_LABELS` (optional) — name → `{filename: class_index}` map that gets
  materialised as `labels.csv` alongside the images, so `data.labels.source: <name>`
  resolves automatically.

## Adding a built-in sample dataset

1. **Register the source files** in `src/raitap/data/samples.py`:

    ```python
    SAMPLE_SOURCES["cifar10_samples"] = [
        ("https://…/airplane.png", "airplane.png"),
        ("https://…/cat.png", "cat.png"),
        # …
    ]
    ```

2. **(Optional) Register labels** if the dataset has a known ground truth that
   matches the model's class space:

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

5. **Update documentation** — add the new sample name to
   `docs/modules/data/own-vs-built-in.md`.

## Sample discovery and label alignment

`data.source` directories are walked **recursively** (`Path.rglob`) and
sample ids are computed as posix-style paths relative to the source root
(e.g. `NORMAL/IM-0001.jpeg`). This supports nested `ImageFolder`-style
layouts where filename stems may collide across class subdirs.

Label alignment is governed by `data.labels.id_strategy`:

- `"auto"` (default) — inspects the id column once. If any value contains
  `/` or `\`, switches to `"relative_path"`; otherwise `"stem"`.
- `"relative_path"` — strips only the file extension during comparison;
  directory components are retained.
- `"stem"` — legacy flat-dir behaviour (`Path(id).stem`).

Both sides (sample ids from disk + label ids from the file) are normalised
the same way before lookup. Duplicate normalised label ids raise a
warning and disable label-based metrics for that run.

## Image preprocessing internals

User-facing surface lives at `docs/modules/data/preprocessing.md` — that doc is the
source of truth for vocabulary exposed to users (no class names, no internal
mechanics). This section covers the implementation so maintainers can extend it
without re-deriving the design.

### Three options

The three `DataConfig.preprocessing` values resolve as follows:

| `origin`           | Config value                  | Resolver                           | Behaviour                                                     |
| ------------------ | ----------------------------- | ---------------------------------- | ------------------------------------------------------------- |
| `off`              | `None` / `""` (absent)        | `_resolve_off`                     | No wrap; warn unless suppressed                               |
| `model-bundled` | `"model-bundled"`          | `_resolve_model_bundled`        | Split `Weights.DEFAULT.transforms()` into shape + value halves |
| `custom-file`      | path to a `.py` file          | `_resolve_custom_file`             | Import file, call `make_preprocessing()` (and optionally `make_data_preprocessing()`); gate on consent |

`origin` matches the literal stored on `ResolvedPreprocessing.origin` —
contributor docs and tests use these names directly so there is no parallel
shorthand to keep in sync.

### Split: data half vs model half

The resolved preprocessing has two halves, one for the data loader and one
for the model boundary:

| Half          | Where                          | What                                  | Why                                                                                                                                   |
| ------------- | ------------------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `data_module` | `Data._load_data`, per image   | Resize + CenterCrop (shape)           | Must run before `_stack_images_numpy` calls `np.stack` so mixed-size directories can be batched. No gradient → safe outside autograd. |
| `model_module`| `Model._apply_preprocessing`   | Normalize (values)                    | Has gradient → must stay inside autograd for Captum / SHAP / torchattacks to attribute and attack on the user-facing `[0, 1]` space.  |

`model-bundled` calls `_split_preset(preset)` for the bundled torchvision
presets (`ImageClassification`, `SemanticSegmentation`). `ObjectDetection`
returns both halves as `None` — detection models normalise internally via
`GeneralizedRCNNTransform`.

`custom-file` defaults to model-side only (`data_module=None`,
`model_module=user_factory()`). Users who need shape preprocessing for
mixed-size folders can additionally export `make_data_preprocessing()` from
their file; `_build_user_factory` imports it with `required=False`. See the
user doc for the recommended split.

`off` returns both halves as `None`.

### Resolver flow

Entry point: `resolve_preprocessing(model_cfg, data_cfg) -> ResolvedPreprocessing`
in `src/raitap/data/preprocessing.py`. The resolver is side-effect-free — it
inspects the config, returns a `ResolvedPreprocessing` record, and leaves
panel rendering / warning emission to its caller.

```
DataConfig.preprocessing
  ├─ None / ""           → _resolve_off              → data_module=None, model_module=None, warnings=[OFF_WARNING] (unless suppressed)
  ├─ "model-bundled"  → _resolve_model_bundled → _split_preset(weights.transforms()) → (Resize+CenterCrop, Normalize)
  └─ <path>.py           → _resolve_custom_file      → data_module=make_data_preprocessing() (optional), model_module=make_preprocessing()
```

Both `Data._load_data` and `Model._apply_preprocessing` call
`resolve_preprocessing` independently and consume their own half. The
resolver runs twice per pipeline; for the `custom-file` option that means
the user file is imported twice. This is intentional — keeping each caller
self-sufficient avoids threading a `ResolvedPreprocessing` through every
signature, and user factories are expected to be cheap/idempotent.

`model-bundled` derives the torchvision arch from `model_cfg.arch`,
falling back to `model_cfg.source` if it names a built-in
(`hasattr(torchvision.models, name)`). Missing-arch lineage raises
`ValueError` with a copy-pasteable fix. `custom-file` enforces `.py`
extension, existence, presence of `make_preprocessing`, and that the factory
returns an `nn.Module`; each failure raises a typed error.

### Wrap insertion

`Model.__init__` calls `_apply_preprocessing(self.backend, config)` in
`src/raitap/models/model.py`. When `resolved.model_module is not None` and
the backend is a `TorchBackend`, the helper moves the module to the
backend's device, switches it to `eval()`, and rebinds:

```python
backend.model = nn.Sequential(model_module, backend.model)
```

`nn.Sequential` keeps the wrap invisible to autograd, captum, shap, and
torchattacks — gradients flow back through the normalize layer, attack
budgets stay defined in the user-facing `[0, 1]` input space.

After wrapping, the helper emits each `resolved.warnings` entry via
`raitap_log.warn` and (for active tiers) a single `raitap_log.info` line
with `resolved.description`. Panel rendering is handled by the standard
log handler.

### Loader integration

`Data._load_data` (`src/raitap/data/data.py`) calls `resolve_preprocessing`,
lifts `resolved.data_module` to a per-image callable via
`module_as_per_image_callable` (which runs in `torch.no_grad` + `eval()`),
and passes it into `_stack_images_numpy(files, per_image_transform=...)`.
The transform runs on each `(C, H, W)` tensor before `np.stack`, so a
directory of mixed-size JPEGs becomes a clean `(N, 3, 224, 224)` batch.

Sample-source paths (`SAMPLE_SOURCES`) get the data half applied to the
already-stacked tensor returned by `_load_sample` — torchvision presets
accept batched tensors so this works identically. Defensive: if `cfg.model`
is absent (minimal test mocks), preprocessing resolution is skipped entirely
and `per_image_transform = None`.

### ONNX path

`_apply_preprocessing` raises `NotImplementedError` if the resolver is
active on an `OnnxBackend` — wrapping a `torch.nn.Module` around an ONNX
session is out of scope for issue #158. The resolver still runs (so the
`off` warning still appears for ONNX-with-no-preprocessing setups), but
`preprocessing: model-bundled` or `preprocessing: ./file.py` on an ONNX
backend short-circuits with a message telling the user to pre-normalize
externally.

### Consent surfaces

The `custom-file` option executes arbitrary Python from a user-named file.
Two consent mechanisms exist, and `_consent_given()` accepts either:

| Path         | Mechanism                                                                                                              |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| CLI          | `--allow-preprocessing-exec` / `-yp` parsed by `_strip_deps_flags` in `src/raitap/deps/bootstrap.py`; on hit, bootstrap sets `os.environ["RAITAP_ALLOW_PREPROCESSING_EXEC"] = "1"` before Hydra |
| Programmatic | `data.acknowledge_preprocessing_exec: bool = False` field on `DataConfig` (`src/raitap/configs/schema.py`)             |

Both are required because `raitap.run(config)` (`src/raitap/api.py`) bypasses
the bootstrap layer entirely — a CLI-only gate would silently allow exec on
the programmatic path. This mirrors `model.allow_unsafe_pickle` (PR #157),
which is also schema-side for the same reason.

The env var is also re-exported through `_exec` so dep-bootstrap re-execs
preserve consent across the sub-process boundary.

### `ResolvedPreprocessing` shape

```python
@dataclass(frozen=True)
class ResolvedPreprocessing:
    data_module: nn.Module | None     # Shape — applied per-image in the loader
    model_module: nn.Module | None    # Value — wrapped at the model boundary
    origin: Literal["off", "model-bundled", "custom-file"]
    description: str
    file_path: Path | None = None     # custom-file only
    file_sha256: str | None = None    # custom-file only
    warnings: list[str] = field(default_factory=list)
```

`is_active` is `True` if either half is non-None.

Downstream consumers:

| Consumer                                           | Reads                                          |
| -------------------------------------------------- | ---------------------------------------------- |
| `Data._load_data`                                  | `data_module` (lifted via `module_as_per_image_callable`) |
| `_apply_preprocessing` wrap + panel + warnings     | `model_module`, `description`, `warnings`      |
| `_mlflow_summary_params` in `mlflow_tracker.py`    | `data.preprocessing` (logged as `data.preprocessing` param) |
| Future HTML report card                            | All fields — same plain phrasing as the panel  |

The `origin` enum value is internal; user-facing renderings translate it
("Off", "Model-bundled (ResNet50 IMAGENET1K_V2)", "Custom file: ./…").

### Test fixture

`src/raitap/data/tests/fixtures/preproc_imagenet.py` is the minimal
custom-file factory exercised by `test_preprocessing.py` and `test_api.py`.
It mirrors the example in `docs/modules/data/preprocessing.md` so the test doubles
as documentation that the example works — keep them in sync when either
changes.
