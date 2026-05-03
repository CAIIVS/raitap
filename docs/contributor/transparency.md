# Contributing to the transparency module

This page describes the internal transparency architecture and how to extend it with new algorithms, frameworks, and visualisers.

## Overview

The transparency module wraps XAI frameworks (Captum, SHAP) behind a unified interface driven by Hydra `_target_` instantiation. Explainers produce an `ExplanationResult` with typed semantics; visualisers validate those semantics before rendering attribution tensors to PNG on disk.

Explainers form a three-level hierarchy (see `src/raitap/transparency/explainers/base_explainer.py` and `full_explainer.py`):

```text
AbstractExplainer                       # root — owns output_payload_kind + check_backend_compat no-op
├── AttributionOnlyExplainer            # you implement compute_attributions(); framework owns explain()
│   ├── CaptumExplainer
│   └── ShapExplainer
└── FullExplainer                       # you implement the full explain() pipeline end-to-end
```

- **`AbstractExplainer`** — root base class. Owns the shared contract: `output_payload_kind` class variable (default `ATTRIBUTIONS`) and the `check_backend_compat` no-op. Never subclass directly.
- **`AttributionOnlyExplainer`** — extend this when the framework should manage the full `explain` pipeline. Subclasses implement only `compute_attributions(model, inputs, **kwargs) → Tensor`; batching, normalisation, result wrapping, and `write_artifacts` are handled by this class.
- **`FullExplainer`** — extend this when you own the entire `explain` pipeline yourself (data conversion, model invocation, result construction, persistence).

Each explainer class sets **`output_payload_kind: ClassVar[ExplanationPayloadKind]`** (default `ATTRIBUTIONS`). `ExplanationResult.semantics` records the payload kind together with scope, method families, sample selection, input metadata, and output-space metadata for downstream validation and reporting.

All visualisers implement `BaseVisualiser`, which defines:

- `visualise(attributions, inputs, **kwargs) -> Figure` (abstract, required)
- `save(attributions, output_path, inputs, **kwargs) -> None` (optional, has default implementation)
- `compatible_algorithms: frozenset[str]` (empty = all algorithms)
- `supported_payload_kinds: ClassVar[frozenset[ExplanationPayloadKind]]` — payload categories the visualiser can render.
- `supported_scopes: ClassVar[frozenset[ExplanationScope]]` — explanation scopes the visualiser can consume, such as local attribution artifacts.
- `supported_output_spaces: ClassVar[frozenset[ExplanationOutputSpace]]` — attribution coordinate spaces the visualiser can render.
- `supported_method_families: ClassVar[frozenset[MethodFamily]]` — method families the visualiser understands.
- `produces_scope: ClassVar[ExplanationScope | None]` — optional produced scope when the visualiser summarizes or otherwise changes the result scope.
- `scope_definition_step: ClassVar[ScopeDefinitionStep | None]` — where the produced scope was defined when `produces_scope` is set.
- `visual_summary: ClassVar[VisualSummarySpec | None]` — optional metadata for summary visualisations.
- `validate_explanation(explanation, attributions, inputs) -> None` — render-time compatibility validation.

Visualisers that preserve the explanation scope leave `produces_scope` unset.
Visualisers that summarize local collections set it explicitly. For example,
SHAP bar, SHAP beeswarm, and tabular bar visualisers consume local tabular or
interpretable attributions and produce cohort visual summaries.

## Important files

The `factory.py` module provides the `Explanation` class and helper functions, which use Hydra's `instantiate()` to build explainers and visualisers from `_target_` keys. Bare class names are automatically resolved to `raitap.transparency.*` paths.

## Runtime flow

Transparency runs after the forward pass in `src/raitap/run/pipeline.py`. For each configured explainer:

1. `Explanation(config, name, model, data)` creates the explainer and visualisers using the factory functions
2. The explainer's `explain()` method returns an `ExplanationResult` (for `AttributionOnlyExplainer`, after calling `compute_attributions()`)
3. `ExplanationResult.write_artifacts()` saves attributions and typed metadata to disk
4. `ExplanationResult.visualise()` iterates through configured visualisers, validates each one against the explanation semantics, calls `visualise()`, and saves the figures

Each explainer writes to its own subdirectory under the Hydra run folder. See {doc}`../using-raitap/understanding-outputs` for more details.

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

    Prefer `AttributionOnlyExplainer` when the library maps to `compute_attributions(model, inputs, ...) -> torch.Tensor`. Otherwise subclass `FullExplainer` and implement `explain(...)` end-to-end.

    Create a new explainer class under `src/raitap/transparency/explainers/`:

    ```python
    # src/raitap/transparency/explainers/new_framework_explainer.py
    from .base_explainer import AttributionOnlyExplainer
    import torch

    class NewFrameworkExplainer(AttributionOnlyExplainer):
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
    from pathlib import Path
    from typing import Any

    import torch
    from matplotlib.figure import Figure

    from raitap.transparency.contracts import (
        ExplanationOutputSpace,
        ExplanationPayloadKind,
        ExplanationScope,
        MethodFamily,
        VisualisationContext,
    )

    from .base_visualiser import BaseVisualiser

    class NewVisualiser(BaseVisualiser):
        # Optional: Restrict to specific algorithms (empty = compatible with all)
        compatible_algorithms: frozenset[str] = frozenset()
        # Declare typed semantic compatibility for the explanation artifacts
        # this visualiser can render.
        supported_payload_kinds = frozenset({ExplanationPayloadKind.ATTRIBUTIONS})
        supported_scopes = frozenset({ExplanationScope.LOCAL})
        supported_output_spaces = frozenset({ExplanationOutputSpace.INPUT_FEATURES})
        supported_method_families = frozenset({MethodFamily.GRADIENT})
        produces_scope = None
        scope_definition_step = None
        visual_summary = None
        
        def __init__(self, **config_kwargs):
            super().__init__()
            # Store configuration

        def visualise(
            self,
            attributions: torch.Tensor,
            inputs: torch.Tensor | None = None,
            *,
            context: VisualisationContext | None = None,
            **kwargs: Any,
        ) -> Figure:
            """Required: Create and return a matplotlib Figure."""
            del context, kwargs
            # Your implementation here
            pass
        
        def save(
            self,
            attributions: torch.Tensor,
            output_path: str | Path,
            inputs: torch.Tensor | None = None,
            *,
            context: VisualisationContext | None = None,
            **kwargs: Any,
        ) -> None:
            """Optional: Override for custom save logic (default uses visualise())."""
            # Default implementation in BaseVisualiser usually sufficient
            super().save(attributions, output_path, inputs, context=context, **kwargs)
    ```

    Reference `src/raitap/transparency/visualisers/captum_visualisers.py` or `shap_visualisers.py` for complete examples.

    Reporting placement comes from the rendered `VisualisationResult.scope`.
    Preserve the input explanation scope for per-sample renderers. Set
    `produces_scope` only when the renderer changes the semantic breadth of the
    result, such as summarizing a local batch into a cohort figure. Do not mark
    representative local montages or arbitrary debug batches as global.

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
