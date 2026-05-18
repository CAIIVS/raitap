---
title: "Image preprocessing"
description: "Most pretrained image models expect their inputs to be resized, center-cropped, and normalized. RAITAP exposes two independent knobs so you can wire each stage where it makes sense."
myst:
  html_meta:
    "description": "Most pretrained image models expect their inputs to be resized, center-cropped, and normalized. RAITAP exposes two independent knobs so you can wire each stage where it makes sense."
---

# Image preprocessing

Pretrained image models usually expect their inputs Resized, CenterCropped,
and Normalized with model-specific mean/std. RAITAP applies nothing by
default — pass unnormalized tensors to an ImageNet-pretrained model and
accuracy silently collapses.

There are **two independent knobs**:

| Knob | Where it runs | Typical contents |
|---|---|---|
| `data.preprocessing` | in the loader, before the batch reaches the model | Resize, CenterCrop (images); feature scaling, encoding (tabular) |
| `data.model_input_transformation` | at the model boundary, every forward pass | Normalize, learnable input scaling |

The split matters: shape changes must run in the loader (for images, per-image
— so mixed-size folders can stack at all) and don't need gradients. The
model-side stage stays inside autograd so Captum/SHAP attribution and PGD/FGSM
attack budgets see the same input space you do.

Both knobs work for any modality. For images, the data-side module is lifted
to a per-image callable and applied as each image is loaded. For tabular
sources, the data-side module is applied once on the stacked `(N, F)` batch.

Each knob accepts the same three values, independently:

- **`null`** (default) — that stage is off.
- **`model-bundled`** — pull the relevant half from the model's bundled
  torchvision preset. Resize+CenterCrop on the data side, Normalize on the
  model side.
- **path to a `.py` file** — load a user-supplied factory decorated with the
  matching RAITAP decorator. Gated by `--allow-preprocessing-exec` / `-yp`
  (CLI) or `acknowledge_preprocessing_exec=True` (Python API).

If both knobs are `null` and inputs are images, a loud warning fires at
startup. If only the model-side knob is `null`, a separate warning fires
(missing Normalize is the silent-metric-corruption case).

## Recipes

### Standard: torchvision model, bundled preprocessing

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

Handles mixed-size folders. Each image becomes 224×224 (for the standard
ImageNet preset), then Normalize runs on every forward pass. This is the
right default whenever `model.source` (or `model.arch`) names a built-in
torchvision model.

`model-bundled` is torchvision-lineage only: it does not read preprocessing
bundled into ONNX exports. For ONNX, use a custom file on the model side.

### Custom Normalize, bundled Resize/Crop

Fine-tuned a torchvision model on non-ImageNet data? Keep the standard
geometry, swap the normalization stats:

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

### Both sides custom

For non-torchvision models, ONNX models, or fully custom pipelines:

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
def normalize_for_model() -> nn.Module:
    return v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
```

The same file can serve both knobs — it's imported and hashed once. Run with
the consent flag:

```{install-tabs}
:uv:
uv run raitap --config-name assessment -yp

:pip:
raitap --config-name assessment -yp
```

### Already preprocessed upstream

Your dataloader emits normalized tensors? Leave both knobs unset and
acknowledge the warning at invocation:

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

The warning is also auto-suppressed for non-image data
(`input_metadata.kind` is `tabular`, `text`, or `time_series`).

When both knobs are off, every image in your directory must already be the
same height and width — the loader stacks them into one batch tensor and
raises if shapes don't match.

## Custom file rules

A factory you decorate must:

- be decorated with `@raitap_preprocessing_factory` (data side) or
  `@raitap_model_input_transformation_factory` (model side),
- take no required arguments,
- return an `nn.Module`.

A file may decorate one factory per side. If you point a knob at a file that
has no matching decorator, RAITAP raises before the model is built. If a
file defines more than one factory for the same side, RAITAP raises with the
duplicate names.

For static analysis, two `Protocol` types are exported:

```python
from raitap.data import DataPreprocessingFactory, ModelInputTransformationFactory

_data_check: DataPreprocessingFactory = resize_and_crop
_model_check: ModelInputTransformationFactory = normalize_for_model
```

RAITAP records each file's path and SHA-256 in tracking metadata so changes
between runs surface in your history.

## Mixed-size folders

The loader stacks images into one batch tensor. Mixed heights/widths only
work if the data-side stage reshapes each image first:

- `data.preprocessing: model-bundled` — handles mixed sizes for any
  torchvision arch (Resize + CenterCrop run per-image).
- `data.preprocessing: ./resize.py` — your factory runs per-image.
- `data.preprocessing: null` — every file must already be the same shape.
