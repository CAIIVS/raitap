# Contributing to the transparency module

This page describes the internal transparency architecture and how to extend it with new algorithms, frameworks, and visualisers.

## Overview

The transparency module wraps XAI frameworks (Captum, SHAP, optional Alibi) behind a unified interface driven by Hydra `_target_` instantiation. Explainers produce an `ExplanationResult`; visualisers render attribution tensors to PNG on disk.

Explainers form a three-level hierarchy (see `src/raitap/transparency/explainers/base_explainer.py` and `full_explainer.py`):

```text
AbstractExplainer                       # root — owns output_payload_kind + check_backend_compat no-op
├── AttributionOnlyExplainer            # you implement compute_attributions(); framework owns explain()
│   ├── CaptumExplainer
│   └── ShapExplainer
└── FullExplainer                       # you implement the full explain() pipeline end-to-end
    └── AlibiExplainer
```

- **`AbstractExplainer`** — root base class. Owns the shared contract: `output_payload_kind` class variable (default `ATTRIBUTIONS`) and the `check_backend_compat` no-op. Never subclass directly.
- **`AttributionOnlyExplainer`** — extend this when the framework should manage the full `explain` pipeline. Subclasses implement only `compute_attributions(model, inputs, **kwargs) → Tensor`; batching, normalisation, result wrapping, and `write_artifacts` are handled by this class.
- **`FullExplainer`** — extend this when you own the entire `explain` pipeline yourself (data conversion, model invocation, result construction, persistence). Used for Alibi, whose API does not map to a simple tensor-in/tensor-out attribution step.

Each explainer class sets **`output_payload_kind: ClassVar[ExplanationPayloadKind]`** (default `ATTRIBUTIONS`). `ExplanationResult` stores `payload_kind` and includes it in `metadata.json`.

All visualisers implement `BaseVisualiser`, which defines:

- `visualise(attributions, inputs, **kwargs) -> Figure` (abstract, required)
- `save(attributions, output_path, inputs, **kwargs) -> None` (optional, has default implementation)
- `compatible_algorithms: frozenset[str]` (empty = all algorithms)
- `supported_payload_kinds: ClassVar[frozenset[ExplanationPayloadKind]]` — default `{ATTRIBUTIONS}`. An **empty** `frozenset()` means the visualiser accepts **all** payload kinds (wildcard). The factory raises `PayloadVisualiserIncompatibilityError` if the explainer’s `output_payload_kind` is not listed when the set is non-empty.
- `report_scope: ClassVar[str]` — defines which report group the visualiser belongs to. The default `"local"` means RAITAP places the output in **Local Explanations** because it represents one sample at a time, as with `CaptumImageVisualiser` or `ShapImageVisualiser`. Set this to `"global"` only when the visualiser itself already produces a true aggregate view for the whole run or dataset, so RAITAP places it in **Global Explanations**. Examples include `ShapBarVisualiser` and `ShapBeeswarmVisualiser`. Representative montages of a few samples are still `"local"`.

After `create_explainer`, `factory.Explanation` may emit **third-party license warnings** (e.g. Alibi BSL) at most once per process via `logging.warning`.

## Important files

The `factory.py` module provides the `Explanation` class and helper functions, which use Hydra's `instantiate()` to build explainers and visualisers from `_target_` keys. Bare class names are automatically resolved to `raitap.transparency.*` paths.

## Runtime flow

Transparency runs after the forward pass in `src/raitap/run/pipeline.py`. For each configured explainer:

1. `Explanation(config, name, model, data)` creates the explainer and visualisers using the factory functions
2. The explainer's `explain()` method returns an `ExplanationResult` (for `AttributionOnlyExplainer`, after calling `compute_attributions()`)
3. `ExplanationResult.write_artifacts()` saves attributions and metadata to disk (`payload_kind` is recorded in metadata)
4. `ExplanationResult.visualise()` iterates through configured visualisers, calling each one's `visualise()` method and saving the figures

Each explainer writes to its own subdirectory under the Hydra run folder. See {doc}`../using-raitap/understanding-outputs` for more details.

## Adding a new algorithm

Captum and SHAP wrappers dispatch to algorithms dynamically via `getattr`, so most new methods require no code changes. Override the algorithm on a specific explainer entry in your transparency config:

```bash
uv run raitap transparency=demo transparency.captum_ig.algorithm=Saliency
uv run raitap transparency=demo transparency.shap_gradient.algorithm=GradientShap
```

Add an integration test to confirm the method works end-to-end. Reference `src/raitap/transparency/explainers/tests/test_captum_explainer.py` for examples.

Some SHAP methods require special init logic. Check `src/raitap/transparency/explainers/shap_explainer.py` for conditionals and add a branch if needed.

## Alibi Explain

- **Class:** `AlibiExplainer` (`FullExplainer`). **Algorithms:** `KernelShap` (PyTorch `nn.Module` black-box, default in `alibi_kernel.yaml`), `TreeShap` (fitted tree-based model — sklearn/XGBoost/LightGBM/CatBoost, pass via `constructor: {tree_model: ...}`), and `IntegratedGradients` (TensorFlow/Keras only — pass `keras_model` in Hydra `constructor`).
- **Licensing:** Alibi is **BSL 1.1**, not GPLv3. See {ref}`Alibi (transparency) <alibi-frameworks>` and the one-time `logging.warning` from `factory._maybe_emit_third_party_license_warnings` when `ALIBI_BSL_LICENSE_WARNING` is true on the explainer class.
- **Installation (this repo):** `uv sync` with `--extra alibi`; the root **`pyproject.toml`** already supplies **`[tool.uv]` overrides**, so you do not add them manually. **Downstream** projects that depend on `raitap[alibi]` must mirror those overrides — see {ref}`Alibi (transparency) <alibi-install-overrides>`.
- **Tests:** `src/raitap/transparency/explainers/tests/test_alibi_explainer.py` uses `needs_alibi` and skips when `alibi` is not installed.

### Alibi algorithms not currently supported

The following Alibi explainers are intentionally absent. This section documents the blockers so future contributors know what needs to change before they can be added.

**`AnchorTabular`, `AnchorImage`, `AnchorText`, `DistributedAnchorTabular`**

These produce *structured* explanations (anchor feature conditions, precision, coverage) rather than attribution tensors. `ExplanationResult` currently requires an `attributions: torch.Tensor` and `write_artifacts()` raises `NotImplementedError` for `ExplanationPayloadKind.STRUCTURED`. Before adding any Anchor method: (1) make `attributions` optional on `ExplanationResult`, (2) implement STRUCTURED persistence (a JSON dump of the anchor result), and (3) fix the hardcoded `attributions.pt` copy in `ExplanationResult.log()`. `AnchorImage` and `AnchorText` also pull in extra-heavy dependencies (image segmentation / spaCy) that do not belong in the `[alibi]` optional extra without deliberate scoping.

**`ALE` (Accumulated Local Effects)**

ALE is a *global*, population-level explanation method. Its output is one effect curve per feature across the dataset — not a tensor shaped like the inputs. It does not fit the per-sample attribution model that `ExplanationResult` expects. Supporting it properly would require a different result type (or a well-defined mapping from global ALE curves to per-sample estimates, which is non-standard).

**`CEM`, `CounterFactual`, `CounterFactualProto`**

All three are TensorFlow/Keras-only and produce counterfactual instances (STRUCTURED output), not attribution tensors. Adding them would require (1) TensorFlow as a hard dependency inside `[alibi]` — a significant architectural choice — and (2) STRUCTURED persistence (same blocker as Anchor methods).

## Adding a new framework

To integrate a new explainability framework:

1. **Implement the wrapper**

    Prefer `AttributionOnlyExplainer` when the library maps to `compute_attributions(model, inputs, ...) -> torch.Tensor`. Otherwise subclass `FullExplainer` and implement `explain(...)` end-to-end (see `alibi_explainer.py`).

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

    from raitap.transparency.contracts import VisualisationContext

    from .base_visualiser import BaseVisualiser

    class NewVisualiser(BaseVisualiser):
        # Optional: Restrict to specific algorithms (empty = compatible with all)
        compatible_algorithms: frozenset[str] = frozenset()
        # Optional: restrict payload kinds (empty frozenset = all kinds)
        # supported_payload_kinds: ClassVar[frozenset[ExplanationPayloadKind]] = frozenset({ExplanationPayloadKind.ATTRIBUTIONS})
        # Optional: set to "global" for true aggregate visualisations.
        report_scope = "local"
        
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

    `report_scope` controls where report builders place the visualisation:
    `"local"` means per-sample output, while `"global"` means true global output.
    Do not mark representative local montages as global. RAITAP can generate its
    own aggregate global summaries from local attribution tensors when a run has
    multiple samples.

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
