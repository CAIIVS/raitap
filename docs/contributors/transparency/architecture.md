# Transparency module architecture

This document explains the design of the transparency module and how its components work together.

## Overview

The transparency module wraps multiple XAI frameworks (Captum, SHAP) behind a unified interface driven by Hydra `_target_` instantiation:

```txt
┌─────────────────────────────────────┐
│           CLI / Config              │
│  transparency=captum                │
│  transparency.algorithm=Saliency    │
└──────────────┬──────────────────────┘
               │
        ┌──────▼──────────┐
        │   explain()     │
        │  (factory.py)   │
        └──────┬──────────┘
               │ Hydra instantiate(_target_)
      ┌────────┴─────────┐
      │                  │
  ┌───▼───────┐   ┌──────▼──────┐
  │ Explainer │   │  Visualiser │
  │ (captum,  │   │  (captum,   │
  │  shap)    │   │   shap)     │
  └───────────┘   └─────────────┘
```

## File structure

```txt
src/raitap/transparency/
├── __init__.py              # Public API
├── factory.py               # explain()
├── methods_registry.py      # VisualiserIncompatibilityError
├── explainers/
│   ├── base.py              # BaseExplainer interface
│   ├── captum_explainer.py  # Wraps all Captum methods
│   └── shap_explainer.py    # Wraps all SHAP methods
└── visualisers/
    ├── base.py                  # BaseVisualiser interface
    ├── captum_visualisers.py    # CaptumImageVisualiser, CaptumTextVisualiser,
    │                            # CaptumTimeSeriesVisualiser
    ├── shap_visualisers.py      # ShapBarVisualiser, ShapBeeswarmVisualiser,
    │                            # ShapForceVisualiser, ShapImageVisualiser,
    │                            # ShapWaterfallVisualiser
    └── tabular_visualiser.py    # TabularBarChartVisualiser
```

## Components

### Factory ([factory.py](../../src/raitap/transparency/factory.py))

Single entry point. Uses Hydra `instantiate()` to build the explainer and visualisers from `_target_` keys in the transparency config:

```python
from raitap.transparency import explain

result = explain(config, model, inputs)
# result["attributions"]   → torch.Tensor
# result["visualisations"] → dict[str, matplotlib.figure.Figure]
# result["run_dir"]        → pathlib.Path
```

Bare class names (no dots) in `_target_` are automatically expanded to their fully-qualified `raitap.transparency.*` paths by `_resolve_target()`.

### Explainers ([explainers/](../../src/raitap/transparency/explainers/))

Compute attributions, return `torch.Tensor`:

```python
class BaseExplainer(ABC):
    @abstractmethod
    def compute_attributions(self, model, inputs, **kwargs) -> torch.Tensor:
        pass
```

Each concrete class wraps one framework and dispatches dynamically via `getattr`:

```python
class CaptumExplainer(BaseExplainer):
    def __init__(self, algorithm: str, **init_kwargs): ...

    def compute_attributions(self, model, inputs, **kwargs) -> torch.Tensor:
        import captum.attr
        method_class = getattr(captum.attr, self.algorithm)
        ...
```

### Visualisers ([visualisers/](../../src/raitap/transparency/visualisers/))

Render attributions to images or charts:

```python
class BaseVisualiser(ABC):
    compatible_algorithms: frozenset[str] = frozenset()

    @abstractmethod
    def visualise(self, attributions, inputs=None, **kwargs) -> Figure: ...

    def save(self, attributions, output_path, inputs=None, **kwargs) -> None: ...
```

`compatible_algorithms` is validated before any computation — a `VisualiserIncompatibilityError` is raised if the chosen algorithm is not in the set.

### Config-driven instantiation

The `_target_` key in a transparency config YAML selects what gets created:

```yaml
# configs/transparency/captum.yaml
_target_: CaptumExplainer
algorithm: IntegratedGradients
visualisers:
  - _target_: CaptumImageVisualiser
```

Selecting a different preset or overriding on the CLI requires no code changes:

```bash
uv run raitap transparency=shap
uv run raitap transparency.algorithm=Saliency
```

## Testing

Tests are organised by component:

- `test_methods.py` — registry correctness
- `test_captum_explainer.py` — Captum wrapper
- `test_shap_explainer.py` — SHAP wrapper
- `test_visualisers.py` — visualiser implementations
- `test_integration.py` — end-to-end workflows
