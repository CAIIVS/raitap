---
title: "Model API"
description: "Reference for model loading and backend abstraction in raitap.models."
---

The public `models` surface is intentionally small: one `Model` class, backed internally by `TorchBackend` and `OnnxBackend` in `src/raitap/models/backend.py`.

## Import

```python
from raitap.models import Model
```

## `Model`

Source: `src/raitap/models/model.py`

```python
class Model(Trackable):
    def __init__(self, config: AppConfig) -> None
    def log(self, tracker: BaseTracker, **kwargs: Any) -> None
```

The constructor resolves `config.model.source` into one of three paths:

- built-in torchvision model name such as `resnet50`
- `.pt` or `.pth` PyTorch asset
- `.onnx` ONNX asset

Constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AppConfig` | — | Run config containing `model.*` and `hardware`. |

Return type: `Model`

Public attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `backend` | `ModelBackend` | Runtime wrapper used for inference and explanation. |

## Supported behaviors

`Model._load_model()` delegates to `_load_from_path()` or `_load_pretrained()` and ultimately produces a backend with a unified runtime interface:

```python
prepared = model.backend._prepare_inputs(inputs)
raw_output = model.backend(prepared)
explainable_model = model.backend.as_model_for_explanation()
```

## Common usage

```python
from raitap.models import Model

model = Model(cfg)
print(model.backend.hardware_label)
```

## Notes on loading

- Torch checkpoints are loaded safely with `torch.load(..., weights_only=True)` first.
- State-dict checkpoints require `model.arch` and `model.num_classes`.
- Pickled `nn.Module` checkpoints still work, but `src/raitap/models/model.py` emits a warning because they are fragile across environments.
- ONNX models are executed through ONNX Runtime and receive device/provider resolution from `src/raitap/models/runtime.py`.

Although `TorchBackend`, `OnnxBackend`, and `ModelBackend` are not re-exported from `raitap.models`, they are central to the behavior of `Model` and are worth understanding when you debug inference or explanation compatibility.

## Example combinations

Built-in pretrained demo model:

```yaml
model:
  source: resnet50
```

State-dict checkpoint:

```yaml
model:
  source: ./weights.pth
  arch: resnet18
  num_classes: 2
  pretrained: false
```

ONNX asset:

```yaml
model:
  source: ./model.onnx
```

Those three cases all flow through the same public `Model` wrapper, which is exactly why the rest of the pipeline can stay format-agnostic.
