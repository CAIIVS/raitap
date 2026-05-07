---
title: "Model And Data Loading"
description: "Understand how RAITAP resolves model assets, runtime backends, datasets, and labels."
---

The `models` and `data` packages are the input boundary of RAITAP. They turn heterogeneous external assets into a predictable pair of tensors: model inference through a `ModelBackend` and data through `Data.tensor`, with optional aligned labels in `Data.labels`.

## What This Concept Solves

Responsible-AI tooling is fragile when model and data loading logic is scattered across scripts. RAITAP isolates that boundary so the rest of the pipeline can assume a small contract: a callable backend, an input tensor, stable sample IDs when available, and optional targets.

This concept feeds directly into `/docs/explainers-and-semantics`, because explainers depend on backend compatibility and input metadata. It also feeds into `/docs/outputs-tracking-and-reporting`, because sample IDs and labels become report metadata and tracking artifacts.

## How It Works Internally

`src/raitap/models/model.py` chooses a load path from `config.model.source`:

- Existing `.onnx` path: create `OnnxBackend.from_path(...)`
- Existing `.pt` or `.pth` path: load TorchScript, a safe state dict, or a pickled `nn.Module`
- Known torchvision name such as `resnet50`: load pretrained weights through `_load_pretrained()`

The backend abstraction lives in `src/raitap/models/backend.py`. `TorchBackend` moves inputs and kwargs onto the correct device and exposes the raw `nn.Module` for explainers. `OnnxBackend` wraps an `onnxruntime.InferenceSession`, normalizes input dtypes, and exposes an `_OnnxExplanationModule` bridge so explainers still get a `torch.nn.Module`-shaped object.

The data path in `src/raitap/data/data.py` uses the same pattern. `Data._load_data()` accepts:

- named demo sets such as `imagenet_samples`
- local image files or directories
- local CSV, TSV, or Parquet files or directories
- URLs that are cached under `~/.cache/raitap`

If labels are configured, `Data._load_labels()` reads a tabular labels file, infers or validates ID columns, converts labels to integer classes, and aligns them against discovered sample IDs. The alignment helpers such as `_resolve_id_strategy()` and `_align_labels_to_samples()` are the reason nested image-folder layouts can still be matched safely.

## Basic Usage

This is the simplest programmatic path: create `Model` and `Data` from a composed config.

```python
from hydra import compose, initialize_config_dir
from pathlib import Path

from raitap.data import Data
from raitap.models import Model

config_dir = Path("src/raitap/configs").resolve()

with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
    cfg = compose(
        config_name="config",
        overrides=[
            "hardware=cpu",
            "model=resnet50",
            "data=imagenet_samples",
            "reporting=disabled",
        ],
    )

model = Model(cfg)
data = Data(cfg)

print(model.backend.hardware_label)
print(data.tensor.shape)
print(data.sample_ids)
```

## Advanced Usage

You can also bypass the full `Data` object and use the public loader helpers directly. This is especially useful for SHAP background datasets.

```python
from raitap.data import load_tensor_from_source, load_numpy_from_source

background = load_tensor_from_source("imagenet_samples", n_samples=2)
tabular = load_numpy_from_source("./features.parquet")

print(background.shape)
print(tabular.shape)
```

<Callout type="warn">`Data._load_data()` does not resize user-supplied image directories. The demo datasets are resized in `src/raitap/data/samples.py`, but consumer image folders are loaded raw and then stacked. If your images have inconsistent shapes, `_stack_images_numpy()` raises a `ValueError` and you need to normalize sizes before loading.</Callout>

<Accordions>
<Accordion title="Why RAITAP keeps backend selection inside Model">
Putting backend resolution in `Model` means the pipeline only needs to reason about one object, but it also means model loading has opinions about supported formats and runtime fallbacks. `src/raitap/models/runtime.py` only accepts `hardware="cpu"` or `hardware="gpu"`, then maps that to CUDA, Intel XPU, or CPU depending on what is available. That is intentionally simpler than exposing every device string from every framework. The trade-off is that low-level runtime tuning belongs in model export and environment setup, not in the RAITAP public API.

```python
from raitap.models.runtime import resolve_torch_device

device = resolve_torch_device("gpu")
print(device)
```
</Accordion>
<Accordion title="Why label alignment uses sample IDs instead of raw row order">
Row-order labels are convenient for toy examples but unsafe for real image folders, especially when files live in nested class directories or when datasets are discovered recursively. `src/raitap/data/data.py` therefore prefers an explicit ID column and normalizes both sides through `_normalise_sample_id()`. That makes nested datasets much safer, but it also means your labels file has to use the same matching strategy as the data source. If you rely on filenames alone in a nested tree, duplicate stems can silently collapse into ambiguous IDs unless you keep `id_strategy=auto` or `relative_path`.

```yaml
data:
  source: ./images
  labels:
    source: ./labels.csv
    id_column: image
    id_strategy: relative_path
```
</Accordion>
</Accordions>

Once the model and data layer has produced stable tensors, the rest of RAITAP becomes a pure orchestration problem.
