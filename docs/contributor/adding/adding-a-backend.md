---
title: "Adding a backend"
description: "How to add a new model backend to RAITAP using the Capability model."
myst:
  html_meta:
    "description": "How to add a new model backend to RAITAP using the Capability model."
---

# Adding a backend

A **backend** wraps a model runtime and exposes a uniform interface to the pipeline. Two backends ship today: `TorchBackend` (autograd-capable) and `OnnxBackend` (forward-only). Add a backend when you need a new runtime (e.g. TensorFlow, TFLite, scikit-learn).

## Steps

### 1. Subclass `ModelBackend`

```python
from raitap.models.backend import ModelBackend
from raitap import backends
from raitap.types import Capability

@backends.register(provides=frozenset({Capability.AUTOGRAD}))
class MyBackend(ModelBackend):
    ...
```

`@backends.register` sets the `provides` class variable at decoration time. It performs a type-check: `provides` must be a `frozenset[Capability]`.

### 2. Implement the abstract methods

`ModelBackend` has three abstract methods you must implement:

```python
@property
@abstractmethod
def hardware_label(self) -> str:
    """Human-readable label for the runtime (e.g. 'CPU', 'CUDA')."""

@abstractmethod
def __call__(self, inputs: torch.Tensor) -> Any:
    """Run inference for ``inputs`` and return raw model output."""

@abstractmethod
def as_model_for_explanation(self) -> nn.Module:
    """Return the nn.Module that explainers consume."""
```

### 3. Declare `provides`

Pass the set of capabilities your backend provides to `@backends.register`:

```python
# Differentiable PyTorch model: provides autograd
@backends.register(provides=frozenset({Capability.AUTOGRAD}))
class TorchBackend(ModelBackend): ...

# Forward-only ONNX runtime: provides nothing
@backends.register(provides=frozenset())
class OnnxBackend(ModelBackend): ...
```

### 4. Optional attributes

Override these class or instance attributes as needed:

| Attribute | Type | Default | Purpose |
|---|---|---|---|
| `expected_input_shape` | `tuple[int \| None, ...] \| None` | `None` | Per-sample input shape. `None` dims = dynamic (batch). |
| `category_names` | `list[str] \| None` | `None` | Class id-to-name table (e.g. from model weights metadata). |
| `task_kind` | `TaskKind` property | `TaskKind.classification` | Task family this backend serves. Override for detection etc. |

### 5. No gate code needed

Do not write compatibility checks in your backend. The shared gate (`AdapterMixin.check_backend_compat`) is inherited by every adapter and raises `BackendIncompatibilityError` automatically when `algorithm.requires - backend.provides` is non-empty. Rule: an algorithm runs on a backend iff `algorithm.requires <= backend.provides`.

## Which capabilities to declare

Most backends provide `{Capability.AUTOGRAD}` (torch) or nothing (forward-only, e.g. ONNX). See {doc}`../capabilities` for the full list, what each means, and which algorithms require it.

## Example: minimal autograd backend

```python
from typing import Any

import torch
from torch import nn

from raitap import backends
from raitap.models.backend import ModelBackend
from raitap.types import Capability


@backends.register(provides=frozenset({Capability.AUTOGRAD}))
class MyTorchBackend(ModelBackend):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @property
    def hardware_label(self) -> str:
        return "CPU"

    def __call__(self, inputs: torch.Tensor) -> Any:
        return self.model(inputs)

    def as_model_for_explanation(self) -> nn.Module:
        return self.model
```
