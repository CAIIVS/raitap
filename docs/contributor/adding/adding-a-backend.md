---
title: "Adding a backend"
description: "How to add a new model backend to RAITAP using the Capability model."
myst:
  html_meta:
    "description": "How to add a new model backend to RAITAP using the Capability model."
---

# Adding a backend

A **backend** wraps a model runtime and exposes a uniform interface to the pipeline. Add a backend when you need a new runtime (e.g. TensorFlow, TFLite, scikit-learn).

## Steps

A backend is a one-file plugin. We will use a fictional backend called `MyBackend`. It depends on Torch, but your real backend might not; the following is just an example.

1. **Subclass `ModelBackend`** and decorate with `@backends.register(provides=..., extensions=...)`.
2. **Declare `provides` and `extensions`**: `provides` is the `frozenset[Capability]` your backend offers; `extensions` is the set of file suffixes it loads. The decorator type-checks both, sets them as class variables, and indexes the backend by extension so model loading resolves the right backend for a given file.
3. **Implement the abstract methods**: `from_path` (construct from a model file), `__call__` (run inference), and the `hardware_label` property.
4. **`predict_callable` is inherited**: it returns `self.__call__`, the universal forward-only shape that model-agnostic explainers consume. You do not implement it.
5. **`autograd_module` is opt-in**: implement it (return the live torch `nn.Module`) and declare `Capability.AUTOGRAD` ONLY if your backend exposes a differentiable torch module. Gradient explainers and attacks get this shape; model-agnostic ones get the predict callable.

```python
from pathlib import Path
from typing import Any

import torch
from torch import nn

from raitap import backends
from raitap.models.backend import ModelBackend
from raitap.types import Capability


@backends.register(provides={Capability.AUTOGRAD}, extensions={".pth", ".pt"})
class MyBackend(ModelBackend):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @classmethod
    def from_path(
        cls, path: Path, *, model_cfg: Any, hardware: str, allow_unsafe_pickle: bool = False
    ) -> ModelBackend:  # load a model file into this backend
        ...

    def __call__(self, inputs: torch.Tensor) -> Any:  # run inference, return raw output
        return self.model(inputs)

    @property
    def hardware_label(self) -> str:  # human-readable runtime label, e.g. "CPU" / "CUDA"
        return get_hardware_label_for_mybackend(self.device)

    def autograd_module(self) -> nn.Module:  # only if AUTOGRAD-capable
        return self.model
```

A non-torch or forward-only backend (e.g. ONNX) declares `provides=FORWARD_ONLY` (the empty capability set, imported from `raitap.types`), skips `autograd_module`, and runs model-agnostic explainers only.

## Which capabilities to declare

Most backends provide `{Capability.AUTOGRAD}` (torch) or `FORWARD_ONLY` (forward-only, e.g. ONNX). See {doc}`../capabilities` for the full list, what each means, and which algorithms require it.

## Optional attributes

Override these class or instance attributes as needed:

| Attribute              | Type                              | Default                   | Purpose                                                      |
| ---------------------- | --------------------------------- | ------------------------- | ------------------------------------------------------------ |
| `expected_input_shape` | `tuple[int \| None, ...] \| None` | `None`                    | Per-sample input shape. `None` dims = dynamic (batch).       |
| `category_names`       | `list[str] \| None`               | `None`                    | Class id-to-name table (e.g. from model weights metadata).   |
| `task_kind`            | `TaskKind` property               | `TaskKind.classification` | Task family this backend serves. Override for detection etc. |

## No gate code needed

Do not write compatibility checks in your backend. The shared gate (`AdapterMixin.check_backend_compat`) is inherited by every adapter and raises `BackendIncompatibilityError` automatically when `algorithm.requires - backend.provides` is non-empty. Rule: an algorithm runs on a backend iff `algorithm.requires <= backend.provides`.
