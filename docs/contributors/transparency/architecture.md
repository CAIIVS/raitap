# Transparency module architecture

This document explains the design of the transparency module and how its components work together.

## Overview

The transparency module wraps multiple XAI frameworks (SHAP, Captum) behind a unified interface:

```txt
┌────────────────────────────────┐
│         User Code              │
│ create_explainer(Captum.IG)    │
└────────────┬───────────────────┘
             │
      ┌──────▼──────────┐
      │  Factory Layer  │
      │   (factory.py)  │
      └──────┬──────────┘
             │
    ┌────────┴─────────┐
    │                  │
┌───▼───────┐   ┌──────▼─────┐
│ Explainer │   │ Visualiser │
│ (captum,  │   │ (image,    │
│  shap)    │   │  tabular)  │
└───────────┘   └────────────┘
```

## Architecture

### File structure

```txt
src/raitap/transparency/
├── __init__.py              # Public API
├── factory.py               # create_explainer(), method_from_config()
├── methods.py               # Registry (Captum, SHAP)
├── explainers/
│   ├── base.py              # BaseExplainer interface
│   ├── captum_explainer.py  # Wraps all Captum methods
│   └── shap_explainer.py    # Wraps all SHAP methods
└── visualisers/
    ├── base.py              # Basevisualiser interface
    ├── image_visualiser.py
    └── tabular_visualiser.py
```

### Components

#### Registry ([methods.py](../../src/raitap/transparency/methods.py))

Declares which methods are supported:

```python
class Captum:
    __framework__ = "captum"
    IntegratedGradients = ExplainerMethod()
    Saliency = ExplainerMethod()
```

#### Explainers ([explainers/](../../src/raitap/transparency/explainers/))

Compute attributions, return `torch.Tensor`:

```python
class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, model, inputs, **kwargs) -> torch.Tensor:
        pass
```

#### Visualisers ([visualisers/](../../src/raitap/transparency/visualisers/))

Render attributions to images/charts:

```python
class Basevisualiser(ABC):
    @abstractmethod
    def save(self, attributions, output_path, **kwargs) -> None:
        pass
```

#### Factory ([factory.py](../../src/raitap/transparency/factory.py))

Creates instances from registry or config:

```python
# Type-safe API
explainer = create_explainer(Captum.IntegratedGradients)

# Config-driven API
method = method_from_config(cfg.transparency)
explainer = create_explainer(method)
```

## Key design patterns

### 1. Registry pattern

Eliminates duplication by capturing attribute names automatically:

```python
class ExplainerMethod:
    def __set_name__(self, owner, name):
        self.framework = owner.__framework__
        self.algorithm = name  # "IntegratedGradients"

class Captum:
    __framework__ = "captum"
    IntegratedGradients = ExplainerMethod()  # Name captured!
```

### 2. Dynamic loading

One wrapper class handles all methods from a framework:

```python
class CaptumExplainer(BaseExplainer):
    def __init__(self, algorithm: str, **init_kwargs):
        self.algorithm = algorithm
        
    def explain(self, model, inputs, **kwargs):
        import captum.attr
        method_class = getattr(captum.attr, self.algorithm)
        method = method_class(model)
        return method.attribute(inputs, **kwargs)
```

### 3. Separation of concerns

Explainers compute attributions → Visualisers render them.

```txt
┌──────────────┐              ┌──────────────┐
│  Explainer   │ torch.Tensor │  Visualiser  │
│   (SHAP)     ├─────────────►│   (Image)    │
└──────────────┘              └──────────────┘
```

## Testing

Tests are organized by component:

- `test_methods.py` - Registry correctness
- `test_captum_explainer.py` - Captum wrapper
- `test_shap_explainer.py` - SHAP wrapper
- `test_visualisers.py` - Visualiser implementations
- `test_integration.py` - End-to-end workflows
