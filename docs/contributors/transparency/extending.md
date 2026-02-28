# Extending RAITAP

This guide shows how to add new explainability methods, frameworks, or visualisers to RAITAP.

## Quick overview

RAITAP's extension system is built around three concepts:

1. **Registry** - Declare available methods in [`methods.py`](../../src/raitap/transparency/methods.py)
2. **Wrapper** - Implement framework-specific logic in [`explainers/`](../../src/raitap/transparency/explainers/)
3. **Factory** - Wire everything together in [`factory.py`](../../src/raitap/transparency/factory.py)

Most extensions only require updating the registry.

## Adding new explainability methods

### Example: adding Captum's `DeepLiftShap`

#### 1. Test it works

1. Write a new test in the framework's test file.

    ```python
    # tests/transparency/test_captum_explainer.py
    def test_deepliftshap(simple_cnn, sample_images):
        explainer = CaptumExplainer("DeepLiftShap")
        attributions = explainer.explain(simple_cnn, sample_images, target=0)
        
        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape == sample_images.shape
    ```

2. Run Pytest.

    ```bash
    uv run pytest tests/transparency/test_captum_explainer.py::test_deepliftshap -v
    ```

#### 2. Add to registry

A method is available to consumers only if it is listed in [the registry](src/raitap/transparency/methods.py).

```python
class Captum:
    __framework__ = "captum"
    
    IntegratedGradients = ExplainerMethod()
    Saliency = ExplainerMethod()
    DeepLiftShap = ExplainerMethod()  # New method added
```

> [!NOTE]
> Some methods require special logic . For instance, some SHAP methods require to be added to the if clause in [shap_explainer.py](src/raitap/transparency/explainers/shap_explainer.py).

#### 3. Test the consumer syntax

Verify that the method truly works for consumers.

```python
from raitap.transparency import create_explainer
from raitap.transparency.methods import Captum

explainer = create_explainer(Captum.DeepLiftShap)
attributions = explainer.explain(model, images, target=0)
```

## Adding a New Framework

As example, we take the integration of OmniXAI.

### 1. Create Wrapper

```python
# src/raitap/transparency/explainers/omnixai_explainer.py
from .base import BaseExplainer
import torch.nn as nn
import torch

class OmniXAIExplainer(BaseExplainer):
    def __init__(self, algorithm: str, **init_kwargs):
        super().__init__()
        self.algorithm = algorithm
        self.init_kwargs = init_kwargs

    def explain(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        target: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        try:
            from omnixai.explainers.vision import VisionExplainer
        except ImportError as e:
            raise ImportError("OmniXAI not installed") from e

        explainer = VisionExplainer(
            model=model,
            explanation_type=self.algorithm,
            **self.init_kwargs
        )
        
        explanations = explainer.explain(inputs, **kwargs)
        
        # Convert to torch.Tensor (adjust based on OmniXAI's format)
        return torch.tensor(explanations.get_explanations())
```

### 2. Create Registry

```python
# src/raitap/transparency/methods.py
class OmniXAI:
    __framework__ = "omnixai"
    
    IntegratedGradient = ExplainerMethod()
    GradientShap = ExplainerMethod()
    LIME = ExplainerMethod()
```

### 3. Update Factory

```python
# src/raitap/transparency/factory.py
from .explainers import BaseExplainer, CaptumExplainer, ShapExplainer, OmniXAIExplainer
from .methods import Captum, SHAP, OmniXAI, ExplainerMethod

def create_explainer(method: ExplainerMethod, modality="image", **init_kwargs):
    if method.framework == "captum":
        explainer = CaptumExplainer(method.algorithm, **init_kwargs)
    elif method.framework == "shap":
        explainer = ShapExplainer(method.algorithm, **init_kwargs)
    elif method.framework == "omnixai":  # ← Add
        explainer = OmniXAIExplainer(method.algorithm, **init_kwargs)
    else:
        raise ValueError(f"Unknown framework: {method.framework}")
    
    explainer._modality = modality
    return explainer

def method_from_config(config) -> ExplainerMethod:
    framework_name = config.framework
    algorithm_name = config.algorithm
    
    if framework_name == "captum":
        return getattr(Captum, algorithm_name)
    elif framework_name == "shap":
        return getattr(SHAP, algorithm_name)
    elif framework_name == "omnixai":  # ← Add
        return getattr(OmniXAI, algorithm_name)
    else:
        raise ValueError(f"Unknown framework: {framework_name}")
```

### 4. Update Schema

```python
# src/raitap/configs/schema.py
class TransparencyFramework(StrEnum):
    captum = "captum"
    shap = "shap"
    omnixai = "omnixai"  # ← Add
```

### 5. Create Config

```yaml
# src/raitap/configs/transparency/omnixai.yaml
framework: omnixai
algorithm: IntegratedGradient
```

### 6. Export

```python
# src/raitap/transparency/__init__.py
from .explainers import OmniXAIExplainer
from .methods import OmniXAI

__all__ = [
    "OmniXAI",
    "OmniXAIExplainer",
    # ...
]
```

## Adding a visualiser

As example, we take text-based feature importance.

### 1. Create visualiser

```python
# src/raitap/transparency/visualisers/text_visualiser.py
from .base import Basevisualiser
import torch
import numpy as np

class Textvisualiser(Basevisualiser):
    def __init__(self, feature_names: list[str] | None = None, top_k: int = 10):
        super().__init__()
        self.feature_names = feature_names
        self.top_k = top_k

    def visualize(self, attributions: torch.Tensor | np.ndarray) -> str:
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.detach().cpu().numpy()
        
        mean_attrs = np.mean(np.abs(attributions), axis=0)
        top_indices = np.argsort(mean_attrs)[-self.top_k:][::-1]
        
        lines = ["Top Feature Importances:", "=" * 40]
        for rank, idx in enumerate(top_indices, 1):
            name = self.feature_names[idx] if self.feature_names else f"Feature_{idx}"
            lines.append(f"{rank:2d}. {name:20s}  {mean_attrs[idx]:8.4f}")
        
        return "\n".join(lines)

    def save(self, attributions: torch.Tensor | np.ndarray, output_path: str, **kwargs):
        text = self.visualize(attributions)
        with open(output_path, "w") as f:
            f.write(text)
```

### 2. Add export

```python
# src/raitap/transparency/visualisers/__init__.py
from .text_visualiser import Textvisualiser

__all__ = [
    "Textvisualiser",
    # ...
]
```

## Adding a Dataset Config

1. Create config file:

    ```yaml
    # src/raitap/configs/data/cifar10.yaml
    name: cifar10
    description: CIFAR-10 - 60k 32x32 color images in 10 classes
    directory: data/cifar10
    num_classes: 10
    image_size: 32
    ```

2. Optionally add loader in *src/raitap/data/loaders.py* for convenience.
