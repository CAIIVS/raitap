---
title: "Transparency Core API"
description: "Reference for explainer factories, result objects, contracts, and public transparency helpers."
---

The transparency package has the widest public surface in RAITAP. The top-level exports are defined in `src/raitap/transparency/__init__.py` and span explainers, results, visualiser factories, semantic helpers, contracts, and domain errors.

## Imports

```python
from raitap.transparency import (
    AbstractExplainer,
    AttributionOnlyExplainer,
    CaptumExplainer,
    ConfiguredVisualiser,
    Explanation,
    ExplanationResult,
    FullExplainer,
    ShapExplainer,
    VisualisationResult,
    check_explainer_visualiser_compat,
    create_explainer,
    create_visualisers,
    explainer_capability,
    infer_input_spec,
    infer_output_space,
    method_families_for_explainer,
)
```

## Explainer classes

### `AbstractExplainer`

Source: `src/raitap/transparency/explainers/base_explainer.py`

```python
class AbstractExplainer:
    output_payload_kind: ClassVar[ExplanationPayloadKind]
    output_scope: ClassVar[ExplanationScope]
    def check_backend_compat(self, backend: object) -> None
```

Base contract for all explainers.

### `AttributionOnlyExplainer`

```python
class AttributionOnlyExplainer(AbstractExplainer, ABC):
    def explain(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        *,
        backend: object | None = None,
        run_dir: str | Path | None = None,
        output_root: str | Path = ".",
        experiment_name: str | None = None,
        explainer_target: str | None = None,
        explainer_name: str | None = None,
        visualisers: list[ConfiguredVisualiser] | None = None,
        raitap_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExplanationResult

    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor
```

Implements batching, metadata normalization, semantic inference, and artifact writing. Subclasses only implement `compute_attributions`.

### `FullExplainer`

Source: `src/raitap/transparency/explainers/full_explainer.py`

```python
class FullExplainer(AbstractExplainer, ABC):
    def explain(...) -> ExplanationResult
```

Use this base class when a framework cannot fit the `compute_attributions()` model.

### `CaptumExplainer`

Source: `src/raitap/transparency/explainers/captum_explainer.py`

```python
class CaptumExplainer(AttributionOnlyExplainer):
    def __init__(self, algorithm: str, **init_kwargs: Any) -> None
    def check_backend_compat(self, backend: object) -> None
    def compute_attributions(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        backend: object | None = None,
        target: int | list[int] | torch.Tensor | None = None,
        baselines: torch.Tensor | None = None,
        **attr_kwargs: Any,
    ) -> torch.Tensor
```

### `ShapExplainer`

Source: `src/raitap/transparency/explainers/shap_explainer.py`

```python
class ShapExplainer(AttributionOnlyExplainer):
    def __init__(self, algorithm: str, **init_kwargs: Any) -> None
    def check_backend_compat(self, backend: object) -> None
    def compute_attributions(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        backend: object | None = None,
        background_data: torch.Tensor | None = None,
        target: int | list[int] | torch.Tensor | None = None,
        **shap_kwargs: Any,
    ) -> torch.Tensor
```

## Factory functions

### `Explanation`

Source: `src/raitap/transparency/factory.py`

```python
class Explanation:
    def __new__(
        cls,
        config: AppConfig,
        explainer_name: str,
        model: Model,
        inputs: torch.Tensor,
        input_metadata: InputSpec | dict[str, Any] | None = None,
        sample_ids: list[str] | None = None,
        sample_names: list[str] | None = None,
        **kwargs: Any,
    ) -> ExplanationResult
```

This is the config-driven explainer entry point used by the pipeline.

### `create_explainer`

```python
def create_explainer(explainer_config: Any) -> tuple[ExplainerAdapter, str]
```

### `create_visualisers`

```python
def create_visualisers(explainer_config: Any) -> list[ConfiguredVisualiser]
```

### `check_explainer_visualiser_compat`

```python
def check_explainer_visualiser_compat(
    explainer_target: str,
    algorithm: str,
    visualisers: list[ConfiguredVisualiser],
) -> None
```

Runs allowlist checks against each configured visualiser.

## Result objects

### `ConfiguredVisualiser`

```python
@dataclass(frozen=True)
class ConfiguredVisualiser:
    visualiser: BaseVisualiser
    call_kwargs: dict[str, Any] = field(default_factory=dict)
```

### `ExplanationResult`

Source: `src/raitap/transparency/results.py`

```python
class ExplanationResult(Trackable):
    def write_artifacts(self) -> None
    def visualise(self, **kwargs: Any) -> list[VisualisationResult]
    def has_visualisations_for_scope(self, scope: ExplanationScope | str) -> bool
    def render_visualisations_for_scope(
        self,
        *,
        scope: ExplanationScope | str,
        sample_index: int | None = None,
    ) -> list[VisualisationResult]
    def log(
        self,
        tracker: BaseTracker | None,
        artifact_path: str = "transparency",
        use_subdirectory: bool = True,
        **kwargs: Any,
    ) -> None
```

### `VisualisationResult`

```python
class VisualisationResult(Trackable):
    def log(
        self,
        tracker: BaseTracker | None,
        artifact_path: str = "transparency",
        use_subdirectory: bool = True,
        **kwargs: Any,
    ) -> None
```

## Semantic helpers

```python
def method_families_for_explainer(explainer: object) -> frozenset[MethodFamily]
def explainer_capability(explainer: object) -> ExplainerCapability
def infer_input_spec(
    inputs: object | None = None,
    *,
    input_metadata: InputSpec | Mapping[str, Any] | None = None,
    kind: InputKind | str | None = None,
    layout: TensorLayout | str | None = None,
    feature_names: Sequence[str] | None = None,
) -> InputSpec
def infer_output_space(...) -> OutputSpaceSpec
```

These helpers are defined in `src/raitap/transparency/semantics.py` and are the backbone of compatibility validation.

## Domain errors

The public error classes are:

- `VisualiserIncompatibilityError`
- `PayloadVisualiserIncompatibilityError`
- `ExplainerBackendIncompatibilityError`

All three are defined under `src/raitap/transparency/exceptions.py`.
