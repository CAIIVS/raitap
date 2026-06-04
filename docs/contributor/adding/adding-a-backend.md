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

Subclass `ModelBackend`, implement three methods, and declare `provides` in the decorator:

```python
from typing import Any

import torch
from torch import nn

from raitap import backends
from raitap.models.backend import ModelBackend
from raitap.types import Capability


@backends.register(provides=frozenset({Capability.AUTOGRAD}))
class MyBackend(ModelBackend):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @property
    def hardware_label(self) -> str:  # human-readable runtime label, e.g. "CPU" / "CUDA"
        return "CPU"

    def __call__(self, inputs: torch.Tensor) -> Any:  # run inference, return raw output
        return self.model(inputs)

    def as_model_for_explanation(self) -> nn.Module:  # the object explainers consume
        return self.model
```

1. **Subclass `ModelBackend`** and decorate with `@backends.register`.
2. **Implement the three abstract methods**: `hardware_label`, `__call__`, `as_model_for_explanation`.
3. **Declare `provides`**: the `frozenset[Capability]` your backend offers. The decorator type-checks it and sets it as a class variable. Torch backends pass `{Capability.AUTOGRAD}`; forward-only runtimes (ONNX) pass `frozenset()`.

## Which capabilities to declare

Most backends provide `{Capability.AUTOGRAD}` (torch) or nothing (forward-only, e.g. ONNX). See {doc}`../capabilities` for the full list, what each means, and which algorithms require it.

## Optional attributes

Override these class or instance attributes as needed:

| Attribute | Type | Default | Purpose |
|---|---|---|---|
| `expected_input_shape` | `tuple[int \| None, ...] \| None` | `None` | Per-sample input shape. `None` dims = dynamic (batch). |
| `category_names` | `list[str] \| None` | `None` | Class id-to-name table (e.g. from model weights metadata). |
| `task_kind` | `TaskKind` property | `TaskKind.classification` | Task family this backend serves. Override for detection etc. |

## No gate code needed

Do not write compatibility checks in your backend. The shared gate (`AdapterMixin.check_backend_compat`) is inherited by every adapter and raises `BackendIncompatibilityError` automatically when `algorithm.requires - backend.provides` is non-empty. Rule: an algorithm runs on a backend iff `algorithm.requires <= backend.provides`.
