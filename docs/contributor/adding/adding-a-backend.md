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
3. **Declare `extra` (and `supported_hardware` if hardware-split)**: `extra` is the uv extra that installs your runtime library (e.g. `"torch"`, `"xgboost"`). `raitap-deps` reads it — import-free, via an AST scan of the decorator — to tell users which extra to install for your file format. Add `supported_hardware={ResolvedHardware.cpu, ...}` only if your library ships a distinct wheel per accelerator; the installable extra is then `f"{extra}-{hw.pyproject_extra_suffix}"` (e.g. `torch-cpu`). Omit it for single-wheel runtimes (the extra is the bare `extra` on all hardware). A file-backed backend without `extra` is invisible to deps inference and falls back to the torch default.
4. **Implement the abstract methods**: `from_path` (construct from a model file), `__call__` (run inference), and the `hardware_label` property.
5. **`predict_callable` is inherited**: it returns `self.__call__`, the universal forward-only shape that model-agnostic explainers consume. You do not implement it.
6. **`autograd_module` is opt-in**: implement it (return the live torch `nn.Module`) and declare `Capability.AUTOGRAD` ONLY if your backend exposes a differentiable torch module. Gradient explainers and attacks get this shape; model-agnostic ones get the predict callable.

```python
from pathlib import Path
from typing import Any

import torch
from torch import nn

from raitap import backends
from raitap.models.backend import ModelBackend
from raitap.types import Capability, ResolvedHardware


@backends.register(
    provides={Capability.AUTOGRAD},
    extensions={".pth", ".pt"},
    extra="mybackend",
    supported_hardware={ResolvedHardware.cpu, ResolvedHardware.cuda},  # ships cpu + cuda wheels
)
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
    def hardware_label(self) -> str:  # free-form display label for the run summary
        return get_hardware_label_for_mybackend(self.device)

    def autograd_module(self) -> nn.Module:  # only if AUTOGRAD-capable
        return self.model
```

A non-torch or forward-only backend (e.g. ONNX) declares `provides=FORWARD_ONLY` (the empty capability set, imported from `raitap.types`), skips `autograd_module`, and runs model-agnostic explainers only.

## Tree / tabular backend

Tree-ensemble runtimes (XGBoost, LightGBM, scikit-learn) follow a separate base class: `TabularTreeBackend`. It owns the torch-to-numpy bridge, the `fitted_estimator()` accessor, and the `(N, C)` probability output shape. Subclass it instead of raw `ModelBackend`.

The concrete subclass implements two methods:

- `from_path`: defer the library import inside the method so the import error surfaces only when the backend is actually used, not at module load. Raise `ImportError` with a pip install hint if the library is absent.
- `_predict_proba`: call the fitted estimator and return an `(N, C)` numpy array of class probabilities.

Register with `provides={Capability.TREE_MODEL, Capability.PREDICT_PROBA}`, a file extension, and `extra="xgboost"`. No `supported_hardware`: XGBoost ships a single wheel (the bare `xgboost` extra on all hardware). The `fitted_estimator()` accessor satisfies the `EstimatorProvider` protocol, which `shap.TreeExplainer` consumes directly. The `predict_callable` method (inherited) returns a callable over the numpy-bridge probabilities, which enables model-agnostic SHAP explainers (e.g. `KernelExplainer`) on tree backends for free.

```python
from pathlib import Path
from typing import Any

import numpy as np

from raitap import backends
from raitap.models.tree_backend import TabularTreeBackend
from raitap.types import Capability


@backends.register(
    provides={Capability.TREE_MODEL, Capability.PREDICT_PROBA},
    extensions={".ubj"},
    extra="xgboost",  # single wheel -> bare extra, no supported_hardware
)
class XGBoostBackend(TabularTreeBackend):
    # __init__(estimator) is inherited from TabularTreeBackend.

    @classmethod
    def from_path(
        cls, path: Path, *, model_cfg: Any, hardware: str, allow_unsafe_pickle: bool = False
    ) -> "XGBoostBackend":
        try:
            import xgboost  # deferred: only required with --extra xgboost
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Run: uv sync --extra xgboost"
            ) from exc
        estimator = xgboost.XGBClassifier()
        estimator.load_model(str(path))
        return cls(estimator)

    def _predict_proba(self, x: np.ndarray) -> np.ndarray:  # (N, C)
        return self._estimator.predict_proba(x)
```

`TabularTreeBackend` inherits `hardware_label` and the CPU/classification defaults, so you do not need to override them unless your runtime supports GPU placement.

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
