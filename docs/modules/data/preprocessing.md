---
title: "Preprocessing"
description: "Configure the two preprocessing stages RAITAP applies between your data and the model: data-side (in the loader) and model-side (at the forward pass)."
myst:
  html_meta:
    "description": "Configure the two preprocessing stages RAITAP applies between your data and the model: data-side (in the loader) and model-side (at the forward pass)."
---

# Preprocessing

This page explains how RAITAP preprocessing works. By default, absolutely no preprocessing is applied; pretrained image models that expect
ImageNet normalization, or tabular models that expect z-scored features, will produce wrong outputs.

The 2 following config keys are available:

| Knob | Where | Typical contents |
|---|---|---|
| `data.preprocessing` | loader, before batch reaches model | Resize + CenterCrop (images), feature scaling (tabular) |
| `data.model_input_transformation` | model boundary, every forward pass. ONLY FOR TORCH MODELS. | Normalize, learnable input layers |

- `preprocessing` runs in the loader so mixed-size image folders can stack at all, and so the work is outside autograd.
- `model_input_transformation` inside autograd so Captum/SHAP attribution and PGD/FGSM attacks operate on the same input space you do.

The following values are allowed for both keys:

- **`null`** (default): no preprocessing
- **`"model-bundled"`**: use the preprocessing bundled inside the model file (e.g. `torchvision` models)
- **path to a `.py` file**: load a user factory decorated with the matching
  RAITAP decorator (see custom examples below). Requires consent — see
  <a href="../../using-raitap/configuration/flags.html#flag-allow-preprocessing-exec"><code>--allow-preprocessing-exec</code></a>.

## Examples

### Torchvision image model, bundled both sides

```{config-tabs}
:yaml:
data:
  source: ./data/images
  preprocessing: model-bundled
  model_input_transformation: model-bundled
model:
  source: resnet50

:python:
from raitap.data import DataConfig, Preprocessing
from raitap.models import ModelConfig

data = DataConfig(
    source="./data/images",
    preprocessing=Preprocessing.model_bundled,
    model_input_transformation=Preprocessing.model_bundled,
)
model = ModelConfig(source="resnet50")
```

Handles mixed-size folders: Resize + CenterCrop run per-image as the loader
stacks the batch; Normalize runs on every forward pass. Works whenever
`model.source` (or `model.arch`) names a built-in torchvision model.
`model-bundled` is Torch-only on both sides — it derives from torchvision
weights lineage, which ONNX exports don't carry. For ONNX, set
`model_input_transformation` to a `.py` path (custom-file model-side
transformation is wired through the ONNX backend's tensor call path).
Data-side preprocessing via `.py` works for ONNX too.

### Tabular model, custom feature scaling

```{config-tabs}
:yaml:
data:
  source: ./data/features.csv
  preprocessing: ./scale.py
  input_metadata:
    kind: tabular

:python:
from raitap.data import DataConfig

data = DataConfig(
    source="./data/features.csv",
    preprocessing="./scale.py",
    input_metadata={"kind": "tabular"},
)
```

```python
# scale.py
import torch
from torch import nn
from raitap.data import raitap_preprocessing_factory


class ZScore(nn.Module):
    def __init__(self, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x):  # x: (N, F)
        return (x - self.mean) / self.std


@raitap_preprocessing_factory
def standardise() -> nn.Module:
    return ZScore(mean=[0.0, 1.0, 2.0], std=[0.5, 0.5, 0.5])
```

The module is invoked once on the stacked `(N, F)` batch — rows are uniform
so per-sample iteration would just waste work. Leave the model-side knob
unset if your network handles its own input layer.

### Bundled Resize/Crop + custom Normalize

Fine-tuned a torchvision arch on non-ImageNet data. Keep the geometry,
swap the stats:

```{config-tabs}
:yaml:
data:
  source: ./data/images
  preprocessing: model-bundled
  model_input_transformation: ./my_normalize.py
model:
  source: resnet50

:python:
from raitap.data import DataConfig, Preprocessing

data = DataConfig(
    source="./data/images",
    preprocessing=Preprocessing.model_bundled,
    model_input_transformation="./my_normalize.py",
)
```

```python
# my_normalize.py
from torch import nn
from torchvision.transforms import v2
from raitap.data import raitap_model_input_transformation_factory


@raitap_model_input_transformation_factory
def normalize_for_my_domain() -> nn.Module:
    return v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
```

### Both sides custom, single file

```{config-tabs}
:yaml:
data:
  source: ./data/images
  preprocessing: ./preprocessing.py
  model_input_transformation: ./preprocessing.py

:python:
from raitap.data import DataConfig

data = DataConfig(
    source="./data/images",
    preprocessing="./preprocessing.py",
    model_input_transformation="./preprocessing.py",
)
```

```python
# preprocessing.py
from torch import nn
from torchvision.transforms import v2
from raitap.data import (
    raitap_model_input_transformation_factory,
    raitap_preprocessing_factory,
)


@raitap_preprocessing_factory
def resize_and_crop() -> nn.Module:
    return nn.Sequential(
        v2.Resize(232, antialias=True),
        v2.CenterCrop(224),
    )


@raitap_model_input_transformation_factory
def normalize() -> nn.Module:
    return v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
```

Same file pointed at both knobs is imported and hashed once. Run with the
consent flag (<a href="../../using-raitap/configuration/flags.html#flag-allow-preprocessing-exec"><code>--allow-preprocessing-exec</code></a>):

```{install-tabs}
:uv:
uv run raitap --config-name assessment -yp

:pip:
raitap --config-name assessment -yp
```

### Already preprocessed upstream

Your dataloader emits normalized tensors? Leave both knobs unset and
acknowledge at invocation (<a href="../../using-raitap/configuration/flags.html#flag-acknowledge-preprocessing-off"><code>--acknowledge-preprocessing-off</code></a>):

::::{tab-set}
:::{tab-item} CLI
```shell
uv run raitap --config-name assessment --acknowledge-preprocessing-off

# Or if installed as a console script:
raitap --config-name assessment --acknowledge-preprocessing-off
```
:::

:::{tab-item} Python API
```python
from raitap import run

run(config, acknowledge_preprocessing_off=True)
```
:::
::::

Non-image kinds (`tabular`, `text`, `time_series` declared via
`input_metadata.kind`) auto-suppress the warning.

## Custom-file rules

A decorated factory must:

- carry `@raitap_preprocessing_factory` (data side) or
  `@raitap_model_input_transformation_factory` (model side),
- take no required arguments,
- return an `nn.Module`.

One factory per side per file. Pointing a knob at a file with no matching
decorator raises before the model is built. Two matching factories raise
with their names.

Two `Protocol` types ship for static analysis:

```python
from raitap.data import DataPreprocessingFactory, ModelInputTransformationFactory

_data_check: DataPreprocessingFactory = resize_and_crop
_model_check: ModelInputTransformationFactory = normalize
```

RAITAP records each file's path and SHA-256 in tracking metadata so changes
between runs surface in your history.

## When does data-side run per-image?

Image sources (`.jpg`, `.png`, …) load per-image — the data-side module is
lifted to a single-image `(C, H, W) → Tensor` callable and applied as each
file is read. That's the only way a directory of varied-size images can be
stacked into one batch.

Tabular sources (`.csv`, `.tsv`, `.parquet`) load as `(N, F)` in one shot —
the data-side module runs once on the full batch.

If you set `data.preprocessing: null` on an image source, every file must
already be the same height and width.
