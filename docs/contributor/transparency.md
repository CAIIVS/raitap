# Contributing to the transparency module

This page describes the internal transparency architecture and how to extend it with new algorithms, frameworks, and visualisers.

## Overview

The transparency module wraps multiple XAI frameworks (Captum, SHAP) behind a unified interface driven by Hydra `_target_` instantiation. Explainers compute attributions using a specific framework and algorithm. Visualisers render those attributions to disk.

All explainers implement `BaseExplainer`, which defines:

- `compute_attributions(model, inputs, **kwargs) -> torch.Tensor` (abstract, required)
- `check_backend_compat(backend) -> None` (optional, for ONNX validation)
- `explain(...)` (implemented in base class, handles batching and orchestration)

All visualisers implement `BaseVisualiser`, which defines:

- `visualise(attributions, inputs, **kwargs) -> Figure` (abstract, required)
- `save(attributions, output_path, inputs, **kwargs) -> None` (optional, has default implementation)
- `compatible_algorithms: frozenset[str]` (class attribute for validation)

## File structure

```text
src/raitap/transparency/
├── __init__.py
├── factory.py
├── results.py
├── exceptions.py
├── algorithm_allowlist.py
├── explainers/
│   ├── __init__.py
│   ├── base_explainer.py
│   ├── captum_explainer.py
│   └── shap_explainer.py
└── visualisers/
    ├── __init__.py
    ├── base_visualiser.py
    ├── captum_visualisers.py
    ├── shap_visualisers.py
    └── tabular_visualiser.py
```

The `factory.py` module provides the `Explanation` class and helper functions, which use Hydra's `instantiate()` to build explainers and visualisers from `_target_` keys. Bare class names are automatically resolved to `raitap.transparency.*` paths.

## Runtime flow

Transparency runs after the forward pass in `src/raitap/run/pipeline.py`. For each configured explainer:

1. `Explanation(config, name, model, data)` creates the explainer and visualisers using the factory functions
2. The explainer's `explain()` method calls `compute_attributions()` internally and returns an `ExplanationResult`
3. `ExplanationResult.write_artifacts()` saves attributions and metadata to disk
4. `ExplanationResult.visualise()` iterates through configured visualisers, calling each one's `visualise()` method and saving the figures

Each explainer writes to its own subdirectory under the Hydra run folder. The structure is:

```text
transparency/
├── explainer1/
│   ├── attributions.pt
│   ├── visualisation1.png
│   └── metadata.json
└── explainer2/
    ├── attributions.pt
    └── metadata.json
```

## Adding a new algorithm

Captum and SHAP wrappers dispatch to algorithms dynamically via `getattr`, so most new methods require no code changes. Override the algorithm on a specific explainer entry in your transparency config:

```bash
uv run raitap transparency=demo transparency.captum_ig.algorithm=Saliency
uv run raitap transparency=demo transparency.shap_gradient.algorithm=GradientShap
```

Add an integration test to confirm the method works end-to-end. Reference `src/raitap/transparency/explainers/tests/test_captum_explainer.py` for examples.

Some SHAP methods require special init logic. Check `src/raitap/transparency/explainers/shap_explainer.py` for conditionals and add a branch if needed.

## Adding a new framework

To integrate a new explainability framework:

1. **Implement the wrapper**

    Create a new explainer class that extends `BaseExplainer` in `src/raitap/transparency/explainers/`:

    ```python
    # src/raitap/transparency/explainers/new_framework_explainer.py
    from .base_explainer import BaseExplainer
    import torch

    class NewFrameworkExplainer(BaseExplainer):
        # Optional: Define ONNX-compatible algorithms
        ONNX_COMPATIBLE_ALGORITHMS: frozenset[str] = frozenset({...})

        def __init__(self, algorithm: str, **init_kwargs):
            super().__init__()
            self.algorithm = algorithm
            self.init_kwargs = init_kwargs

        def check_backend_compat(self, backend: object) -> None:
            """Optional: Validate algorithm compatibility with backend (e.g., ONNX)."""
            # See CaptumExplainer for reference implementation
            pass

        def compute_attributions(
            self,
            model: torch.nn.Module,
            inputs: torch.Tensor,
            backend: object | None = None,
            **kwargs
        ) -> torch.Tensor:
            """Required: Compute attributions and return torch.Tensor."""
            # Your implementation here
            pass
    ```

    Reference `src/raitap/transparency/explainers/captum_explainer.py` or `shap_explainer.py` for complete examples.

2. **Export from `__init__.py`**

    Export the class from both the explainers package and the top-level transparency package:

    ```python
    # src/raitap/transparency/explainers/__init__.py
    from .new_framework_explainer import NewFrameworkExplainer

    __all__ = [..., "NewFrameworkExplainer"]
    ```

    ```python
    # src/raitap/transparency/__init__.py
    from .explainers import NewFrameworkExplainer

    __all__ = [..., "NewFrameworkExplainer"]
    ```

3. **Create a config preset**

    Add a config file under `src/raitap/configs/transparency/`:

    ```yaml
    # src/raitap/configs/transparency/new_framework.yaml
    my_explainer:
      _target_: NewFrameworkExplainer
      algorithm: SomeAlgorithm
      visualisers:
        - _target_: CaptumImageVisualiser
    ```

4. **Use it**

    ```bash
    uv run raitap transparency=new_framework
    uv run raitap transparency=new_framework transparency.my_explainer.algorithm=AnotherAlgorithm
    ```

5. **Update documentation**

    Add the new framework to `docs/modules/transparency/frameworks-and-libraries.md` with supported algorithms, ONNX compatibility notes, and visualiser compatibility.

## Adding a visualiser

To add a new visualiser:

1. **Implement the visualiser**

    Create a new visualiser class that extends `BaseVisualiser` in `src/raitap/transparency/visualisers/`:

    ```python
    # src/raitap/transparency/visualisers/new_visualiser.py
    from .base_visualiser import BaseVisualiser
    import torch
    from matplotlib.figure import Figure

    class NewVisualiser(BaseVisualiser):
        # Optional: Restrict to specific algorithms (empty = compatible with all)
        compatible_algorithms: frozenset[str] = frozenset()
        
        def __init__(self, **config_kwargs):
            super().__init__()
            # Store configuration

        def visualise(
            self,
            attributions: torch.Tensor,
            inputs: torch.Tensor | None = None,
            **kwargs
        ) -> Figure:
            """Required: Create and return a matplotlib Figure."""
            # Your implementation here
            pass
        
        def save(
            self,
            attributions: torch.Tensor,
            output_path: str | Path,
            inputs: torch.Tensor | None = None,
            **kwargs
        ) -> None:
            """Optional: Override for custom save logic (default uses visualise())."""
            # Default implementation in BaseVisualiser usually sufficient
            super().save(attributions, output_path, inputs, **kwargs)
    ```

    Reference `src/raitap/transparency/visualisers/captum_visualisers.py` or `shap_visualisers.py` for complete examples.

2. **Export from `__init__.py`**

    Export the class from both the visualisers package and the top-level transparency package:

    ```python
    # src/raitap/transparency/visualisers/__init__.py
    from .new_visualiser import NewVisualiser

    __all__ = [..., "NewVisualiser"]
    ```

    ```python
    # src/raitap/transparency/__init__.py
    from .visualisers import NewVisualiser

    __all__ = [..., "NewVisualiser"]
    ```

3. **Use it**

    Override the visualisers list on the CLI:

    ```bash
    uv run raitap "transparency.my_explainer.visualisers=[{_target_: NewVisualiser}]"
    ```

    Or embed it in a custom transparency config:

    ```yaml
    # src/raitap/configs/transparency/custom.yaml
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    visualisers:
      - _target_: NewVisualiser
    ```

    ```bash
    uv run raitap transparency=custom
    ```

## Extension points

To add dataset configs, see the data module contributor documentation. Dataset handling is separate from transparency.
