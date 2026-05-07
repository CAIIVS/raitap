---
title: "Transparency Visualiser API"
description: "Reference for the built-in visualiser classes exported by raitap.transparency."
---

RAITAP exports nine visualiser classes. They are all defined under `src/raitap/transparency/visualisers/` and share the `BaseVisualiser` interface.

## Base contract

```python
class BaseVisualiser(ABC):
    def validate_explanation(
        self,
        explanation: object,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None,
    ) -> None
    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> Figure
    def save(
        self,
        attributions: torch.Tensor,
        output_path: str | Path,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        **kwargs: Any,
    ) -> None
```

## Captum visualisers

Source: `src/raitap/transparency/visualisers/captum_visualisers.py`

### `CaptumImageVisualiser`

```python
def __init__(
    self,
    method: str = "blended_heat_map",
    sign: str = "all",
    show_colorbar: bool = True,
    title: str | None = None,
    include_original_image: bool = True,
) -> None
```

Renders image attributions using Captum's image visualization helpers.

### `CaptumTimeSeriesVisualiser`

```python
def __init__(
    self,
    method: str = "overlay_individual",
    sign: str = "absolute_value",
) -> None
```

Renders `(T, C)` or `(B, T, C)` time-series attributions.

### `CaptumTextVisualiser`

```python
def visualise(
    self,
    attributions: torch.Tensor,
    inputs: torch.Tensor | None = None,
    token_labels: list[str] | None = None,
    **kwargs: Any,
) -> Figure
```

Uses a Matplotlib horizontal bar chart instead of Captum's HTML text output.

## SHAP visualisers

Source: `src/raitap/transparency/visualisers/shap_visualisers.py`

### `ShapBarVisualiser`

```python
def __init__(self, feature_names: list[str] | None = None, max_display: int = 20) -> None
```

### `ShapBeeswarmVisualiser`

```python
def __init__(self, feature_names: list[str] | None = None, max_display: int = 20) -> None
```

### `ShapWaterfallVisualiser`

```python
def __init__(
    self,
    feature_names: list[str] | None = None,
    expected_value: float = 0.0,
    sample_index: int = 0,
    max_display: int = 10,
) -> None
```

### `ShapForceVisualiser`

```python
def __init__(
    self,
    feature_names: list[str] | None = None,
    expected_value: float = 0.0,
    sample_index: int = 0,
) -> None
```

### `ShapImageVisualiser`

```python
def __init__(
    self,
    max_samples: int = 4,
    title: str | None = None,
    include_original_image: bool = True,
    show_colorbar: bool = True,
    cmap: str = "coolwarm",
    overlay_alpha: float = 0.65,
) -> None
```

Only compatible with `GradientExplainer` and `DeepExplainer`.

## Framework-agnostic visualiser

Source: `src/raitap/transparency/visualisers/tabular_visualiser.py`

### `TabularBarChartVisualiser`

```python
def __init__(self, feature_names: list[str] | None = None) -> None
```

Produces a cohort-level summary from local tabular attributions by plotting mean absolute feature attribution.

## Common usage pattern

Visualisers are almost always created by config:

```yaml
visualisers:
  - _target_: CaptumImageVisualiser
  - _target_: ShapBarVisualiser
    constructor:
      max_display: 15
```

But direct usage is still possible:

```python
from raitap.transparency import CaptumImageVisualiser

visualiser = CaptumImageVisualiser(show_colorbar=False)
fig = visualiser.visualise(attributions inputs=inputs)
```

Use the visualiser-specific constructor to tune layout, but rely on `ExplanationResult.visualise()` when you want full semantic validation and artifact persistence.
