# Archived: Extending RAITAP

This page is preserved from the pre-Sphinx documentation set and is kept as source
material for a later rewrite.

# Extending RAITAP

This guide shows how to add new explainability methods, frameworks, visualisers, or datasets to RAITAP.

## Adding a new algorithm to an existing framework

Captum and SHAP wrappers dispatch to algorithms dynamically via `getattr`, so most new methods require no code changes — just use the algorithm name directly on the CLI:

```bash
uv run raitap transparency.algorithm=Saliency
uv run raitap transparency.algorithm=GradientShap
```

Add an integration test to confirm the method works end-to-end.

### Captum

```python
# tests/transparency/test_captum_explainer.py
def test_deepliftshap(simple_cnn, sample_images):
    explainer = CaptumExplainer("DeepLiftShap")
    attributions = explainer.compute_attributions(simple_cnn, sample_images, target=0)

    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == sample_images.shape
```

### SHAP

Some SHAP methods require special init logic. Check [`shap_explainer.py`](../../src/raitap/transparency/explainers/shap_explainer.py) for any conditionals; add a branch if needed, then test:

```python
# tests/transparency/test_shap_explainer.py
def test_kernel_explainer(simple_cnn, sample_images):
    explainer = ShapExplainer("KernelExplainer")
    attributions = explainer.compute_attributions(simple_cnn, sample_images)

    assert isinstance(attributions, torch.Tensor)
```

## Adding a new framework

As an example, we integrate OmniXAI.

### 1. Implement the wrapper

```python
# src/raitap/transparency/explainers/omnixai_explainer.py
from .base_explainer import BaseVisplainer
import torch

class OmniXAIExplainer(BaseExplainer):
    def __init__(self, algorithm: str, **init_kwargs):
        super().__init__()
        self.algorithm = algorithm
        self.init_kwargs = init_kwargs

    def compute_attributions(self, model, inputs, **kwargs) -> torch.Tensor:
        try:
            from omnixai.explainers.vision import VisionExplainer
        except ImportError as e:
            raise ImportError("OmniXAI not installed. Run: uv add omnixai") from e

        explainer = VisionExplainer(
            model=model, explanation_type=self.algorithm, **self.init_kwargs
        )
        explanations = explainer.explain(inputs, **kwargs)
        return torch.tensor(explanations.get_explanations())
```

### 2. Export from `__init__.py`

```python
# src/raitap/transparency/explainers/__init__.py
from .omnixai_explainer import OmniXAIExplainer

__all__ = [..., "OmniXAIExplainer"]
```

Export from the top-level package too:

```python
# src/raitap/transparency/__init__.py
from .explainers import OmniXAIExplainer

__all__ = [..., "OmniXAIExplainer"]
```

### 3. Create a transparency config

```yaml
# src/raitap/configs/transparency/omnixai.yaml
_target_: OmniXAIExplainer
algorithm: IntegratedGradient
visualisers:
  - _target_: CaptumImageVisualiser
```

### 4. Use it

```bash
uv run raitap transparency=omnixai
uv run raitap transparency=omnixai transparency.algorithm=LIME
```

## Adding a visualiser

As an example, we add a text-based feature importance visualiser.

### 1. Implement the visualiser

```python
# src/raitap/transparency/visualisers/text_visualiser.py
from .base_visualiser import BaseVisualiser
import torch
import numpy as np

class TextVisualiser(BaseVisualiser):
    def __init__(self, feature_names: list[str] | None = None, top_k: int = 10):
        super().__init__()
        self.feature_names = feature_names
        self.top_k = top_k

    def visualise(self, attributions: torch.Tensor, **kwargs):
        attrs = attributions.detach().cpu().numpy()
        mean_attrs = np.mean(np.abs(attrs), axis=0)
        top = np.argsort(mean_attrs)[-self.top_k:][::-1]
        lines = [
            f"{self.feature_names[i] if self.feature_names else f'Feature_{i}'}: {mean_attrs[i]:.4f}"
            for i in top
        ]
        return "\n".join(lines)

    def save(self, attributions, output_path, **kwargs):
        with open(output_path, "w") as f:
            f.write(self.visualise(attributions))
```

### 2. Export from `__init__.py`

```python
# src/raitap/transparency/visualisers/__init__.py
from .text_visualiser import TextVisualiser

__all__ = [..., "TextVisualiser"]
```

Export from the top-level package too:

```python
# src/raitap/transparency/__init__.py
from .visualisers import TextVisualiser

__all__ = [..., "TextVisualiser"]
```

### 3. Use it

Override the visualisers list on the CLI (Hydra list syntax):

```bash
uv run raitap "transparency.visualisers=[{_target_: TextVisualiser}]"
```

Or embed it in a custom transparency config:

```yaml
# src/raitap/configs/transparency/captum_text.yaml
_target_: CaptumExplainer
algorithm: IntegratedGradients
visualisers:
  - _target_: TextVisualiser
```

```bash
uv run raitap transparency=captum_text
```

## Adding a built-in dataset config

1. Create the config file:

    ```yaml
    # src/raitap/configs/data/cifar10.yaml
    name: cifar10
    description: CIFAR-10 — 60k 32×32 colour images in 10 classes
    source: cifar10
    ```

2. Optionally add a loader in [`src/raitap/data/loader.py`](../../src/raitap/data/loader.py) so `load_data("cifar10")` works.

3. Use it:

    ```bash
    uv run raitap data=cifar10
    ```
