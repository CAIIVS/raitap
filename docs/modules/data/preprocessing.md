---
title: "Image preprocessing"
description: "Most pretrained image models expect their inputs to be normalized (resized, center-cropped, and rescaled with a model-specific mean and standard deviation)."
myst:
  html_meta:
    "description": "Most pretrained image models expect their inputs to be normalized (resized, center-cropped, and rescaled with a model-specific mean and standard deviation)."
---

# Image preprocessing

This page describes how to configure the image preprocessing RAITAP applies
before each forward pass.

Most pretrained image models expect their inputs to be normalized (resized,
center-cropped, and rescaled with a model-specific mean and standard
deviation). RAITAP does **not** preprocess your images by default — if you
hand a pretrained model raw `[0, 1]` tensors when it was trained on
ImageNet-normalized inputs, accuracy can silently collapse. The
`data.preprocessing` option lets you pick which preprocessing RAITAP applies
before the model sees a sample.

There are three options. Pick one.

## Option 1: Off (default)

```{config-tabs}
:yaml:
data:
  source: ./data/images
  # no `preprocessing` key

:python:
from raitap.data import DataConfig

data = DataConfig(source="./data/images")  # preprocessing defaults to None
```

No preprocessing is applied; RAITAP forwards your images to the model
unchanged. At startup you see a loud warning telling you metrics may be
incorrect.

**Use this when** you have already preprocessed your images upstream (e.g.
your dataloader emits normalized tensors), or when your data is not images at
all (tabular, time-series — the warning is auto-suppressed for these).

To silence the warning for an already-preprocessed image dataset, pass the
acknowledgement at invocation time — it is not a config-file option.

::::{tab-set}
:::{tab-item} CLI
```shell
uv run raitap --config-name assessment --acknowledge-preprocessing-off

# If RAITAP is installed as a console script:
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

**Requirement:** every image in your directory must already be the same
height and width. The loader stacks them into a single batch tensor and
raises if shapes don't match. If your folder has mixed-size images, pick
option 2 or option 3.

## Option 2: The model's bundled preprocessing

```{config-tabs}
:yaml:
data:
  source: ./data/images
  preprocessing: model-bundled
model:
  source: resnet50          # or any other built-in torchvision model

:python:
from raitap.data import DataConfig, Preprocessing
from raitap.models import ModelConfig

data = DataConfig(
    source="./data/images",
    preprocessing=Preprocessing.model_bundled,
)
model = ModelConfig(source="resnet50")  # or any other built-in torchvision model
```

RAITAP looks up the preprocessing that ships with the model's pretrained
weights (Resize, CenterCrop, and Normalize with the mean/std the model was
trained with) and applies it before every forward pass.

**Use this when** your model is a standard torchvision model. This is the
right default for `--demo` and for any config where `model.source` names a
built-in torchvision model.

**Handles mixed-size folders.** Resize and CenterCrop run per-image as the
loader stacks the batch, so a directory of varied-size JPEGs loads
successfully (each image becomes 224×224 for the standard ImageNet preset).
The Normalize step then runs at the model boundary on every forward pass.

**Requirements:** either `model.arch` is set (e.g. `arch: resnet50`) or
`model.source` is the name of a built-in torchvision model. Option 2 is
torchvision-lineage only: it does not read preprocessing bundled into ONNX
exports, and it is unsupported for ONNX models or model files with no
torchvision lineage. Use option 3 instead.

## Option 3: Your own preprocessing file

```{config-tabs}
:yaml:
data:
  source: ./data/images
  preprocessing: ./preprocessing.py

:python:
from raitap.data import DataConfig

data = DataConfig(
    source="./data/images",
    preprocessing="./preprocessing.py",
)
```

Then re-run with the consent flag (see the [Consent gate](#consent-gate)
section below):

```{install-tabs}
:uv:
uv run raitap --config-name assessment --allow-preprocessing-exec

:pip:
raitap --config-name assessment --allow-preprocessing-exec
```

RAITAP loads your Python file and calls its `make_preprocessing()` factory.
The returned module is applied before every forward pass.

**Use this when** you need non-standard preprocessing — custom mean/std,
non-ImageNet inputs, a different crop size, extra steps, or a model whose
bundled preprocessing is unavailable.

For ONNX models, option 3 participates in RAITAP's normal tensor/model call
path: the Python preprocessing module runs before RAITAP calls the ONNX
backend. The low-level `OnnxBackend.forward_numpy(...)` API remains raw and
does not apply Python preprocessing on its own.

### Consent gate

Loading `./preprocessing.py` executes arbitrary Python code from disk, so
RAITAP refuses unless you opt in. Choose one:

::::{tab-set}
:::{tab-item} CLI
```shell
uv run raitap --config-name assessment -yp

# If RAITAP is installed as a console script:
raitap --config-name assessment -yp
```
:::

:::{tab-item} Python API
```python
from raitap import run

run(config, acknowledge_preprocessing_exec=True)
```
:::
::::

Without either, RAITAP refuses with a message pointing you back here.

### What your file must look like

Create `preprocessing.py` next to your config and export a
`make_preprocessing()` function that returns an `nn.Module`:

```python
from torch import nn
from torchvision.transforms import v2


def make_preprocessing() -> nn.Module:
    return nn.Sequential(
        v2.Resize(232, antialias=True),
        v2.CenterCrop(224),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    )
```

The factory must:

- Be named `make_preprocessing` (no arguments).
- Return an `nn.Module` — typically an `nn.Sequential` of torchvision
  `v2.*` transforms. `Resize`, `CenterCrop`, and `Normalize` are the usual
  ingredients; the mean/std above are ImageNet values, replace them with
  your dataset's statistics if you trained on something else.

The contract is also exposed as a `Protocol` so type checkers (Pyright,
mypy) flag arity or return-type mistakes before the pipeline runs:

```python
from raitap.data import PreprocessingFactory

_check: PreprocessingFactory = make_preprocessing
```

At runtime RAITAP enforces the same contract: a factory that declares
required positional/keyword args, or returns something other than an
`nn.Module`, raises `TypeError` before the module is wrapped.

The example above reproduces standard ImageNet preprocessing — equivalent to
option 2 for any ImageNet-pretrained model. Adapt it (different crop size,
your own mean/std, extra augmentations turned off at eval time) to fit your
model.

RAITAP records the path and a content hash of your file so changes between
runs show up in your tracking history.

### Mixed-size folders with option 3

Option 3 by default runs the whole `make_preprocessing()` module at the model
boundary on a pre-stacked batch — which means every image in your folder must
already be the same height and width by the time the loader stacks them. If
your folder has mixed-size images, do one of:

- **Switch to option 2** — it handles mixed sizes for you (recommended for
  ImageNet-style models).
- **Pre-resize externally** — run a one-off script that resizes all images to
  a uniform shape.
- **Export a second factory** — alongside `make_preprocessing` you may also
  export `make_data_preprocessing()`, which returns an `nn.Module` that runs
  per-image during loading (before the batch is stacked). Put Resize and
  CenterCrop in this factory, and leave Normalize in `make_preprocessing`:

  ```python
  def make_data_preprocessing() -> nn.Module:
      return nn.Sequential(
          v2.Resize(232, antialias=True),
          v2.CenterCrop(224),
      )


  def make_preprocessing() -> nn.Module:
      return v2.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
      )
  ```

  This mirrors what option 2 does internally — shape changes happen in the
  loader, value changes happen at the model boundary so gradients (for
  Captum, SHAP) and attack budgets (for PGD, FGSM) all stay correct.
