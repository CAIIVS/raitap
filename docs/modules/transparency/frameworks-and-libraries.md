---
title: "Supported libraries"
description: "Explainers support three config buckets:"
myst:
  html_meta:
    "description": "Explainers support three config buckets:"
---

# Supported libraries

## `constructor`, `call`, and `raitap` keys

Explainers support three config buckets:

- `constructor`: kwargs for the explainer constructor or underlying library object
- `call`: verbatim library kwargs for the underlying attribution call
- `raitap`: RAITAP-owned runtime options such as batching, progress bars, and sample-name metadata

Visualisers continue to support `constructor` and `call` only.

This keeps the boundary clear for users: `call` is what Captum or SHAP sees, while `raitap` is what RAITAP itself consumes. Example:

```{config-tabs}
:yaml:
transparency:
  my_first_explainer:
    _target_: "ShapExplainer"
    algorithm: "GradientExplainer"
    constructor:
      local_smoothing: 0.0
    call:
      target: 0
      background_data:
        source: imagenet_samples
    raitap:
      batch_size: 1
    visualisers:
      - _target_: "ShapImageVisualiser"
        call:
          max_samples: 1

:python:
from raitap.transparency import shap, shap_image

transparency = {
    "my_first_explainer": shap(
        algorithm="GradientExplainer",
        constructor={"local_smoothing": 0.0},
        call={
            "target": 0,
            "background_data": {"source": "imagenet_samples"},
        },
        raitap={"batch_size": 1},
        visualisers=[shap_image(call={"max_samples": 1})],
    ),
}
```

## Typed semantics and visualiser compatibility

RAITAP uses typed scope, method-family, and output-space semantics to validate
visualisers against the explanation artifact they receive. In short:

- explainers produce typed `ExplanationResult.semantics`
- visualisers declare the payload kinds, scopes, output spaces, and method
  families they can render
- reporting places rendered figures by `VisualisationResult.scope`; see
  {doc}`../reporting/output` for report section placement

| Visualiser | Consumes | Produces | Notes |
| --- | --- | --- | --- |
| `CaptumImageVisualiser` | Local image input-feature attributions and supported image/CAM spatial maps | Local visualisation | Rejects tabular, token, and time-series layouts. |
| `CaptumTextVisualiser` | Local token-sequence attributions | Local visualisation | Requires explicit token metadata. |
| `CaptumTimeSeriesVisualiser` | Local time-series attributions | Local visualisation | Requires explicit time-series metadata. |
| `ShapImageVisualiser` | Local image-shaped SHAP values from `GradientExplainer` or `DeepExplainer` | Local visualisation | Intended for pixel-level SHAP values, not tabular SHAP outputs. |
| `ShapBarVisualiser` | Local tabular or interpretable SHAP attributions | Aggregated visual summary | Summarizes the selected batch, so it is reported under Aggregated Explanations. |
| `ShapBeeswarmVisualiser` | Local tabular or interpretable SHAP attributions | Aggregated visual summary | Summarizes attribution distributions for the selected batch. |
| `ShapForceVisualiser` | Local tabular or interpretable SHAP attributions for one selected sample | Local visualisation | Preserves local scope. |
| `ShapWaterfallVisualiser` | Local tabular or interpretable SHAP attributions for one selected sample | Local visualisation | Preserves local scope. |
| `TabularBarChartVisualiser` | Local tabular or interpretable attributions | Aggregated visual summary | Uses mean absolute attribution-style aggregation for the selected batch. |

See {doc}`visualisers` for per-visualiser previews, constructor kwargs, and
modality constraints. Contributor-facing details about semantic contracts are
documented in {doc}`../../contributor/transparency`.

## Explainer libraries

### Captum

#### Docs

- [Explainers](https://captum.ai/api/)
- [Visualisers](https://captum.ai/api/utilities.html#visualization)

#### Explainers

`CaptumExplainer` gives access to [all Captum explainers](https://captum.ai/api/).

```{config-tabs}
:yaml:
transparency:
  my_captum_explainer:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    constructor: {}
    call:
      target: 0

:python:
from raitap.transparency import captum

transparency = {
    "my_captum_explainer": captum(
        algorithm="IntegratedGradients",
        call={"target": 0},
    ),
}
```

#### ONNX compatibility

Only algorithms that do not depend on Torch `autograd` are compatible:

- `FeatureAblation`
- `FeaturePermutation`
- `Occlusion`
- `ShapleyValueSampling`
- `ShapleyValues`
- `KernelShap`
- `Lime`

#### Visualiser compatibility

RAITAP currently supports the following [Captum visualisers](https://captum.ai/api/utilities.html#visualization).

- `CaptumImageVisualiser`
- `CaptumTextVisualiser`
- `CaptumTimeSeriesVisualiser`

Compatibility is semantic, not just framework-based. Image visualisation expects
image input metadata or supported CAM/spatial-map output. Text and time-series
visualisation require explicit token or time-series metadata before they can be
validated.

### SHAP

#### Docs

- [Explainers](https://shap.readthedocs.io/en/latest/api.html#explainers)
- [Visualisers](https://shap.readthedocs.io/en/latest/api.html#plots)

#### Explainers

`ShapExplainer` gives access to [all SHAP explainers](https://shap.readthedocs.io/en/latest/api.html#explainers).

```{config-tabs}
:yaml:
transparency:
  my_shap_explainer:
    _target_: ShapExplainer
    algorithm: GradientExplainer
    constructor: {}
    call:
      target: 0
      background_data:
        source: imagenet_samples
    raitap:
      batch_size: 1

:python:
from raitap.transparency import shap

transparency = {
    "my_shap_explainer": shap(
        algorithm="GradientExplainer",
        call={
            "target": 0,
            "background_data": {"source": "imagenet_samples"},
        },
        raitap={"batch_size": 1},
    ),
}
```

`GradientExplainer`, `DeepExplainer`, and `KernelExplainer` usually require
`background_data`. If it is not provided, RAITAP falls back to the input batch.

`DeepExplainer` can fail on PyTorch models that use `SiLU` activations (for example EfficientNet variants) due to autograd/in-place limitations. In those cases, use `GradientExplainer`.

#### ONNX compatibility

Only `KernelExplainer` is compatible.

#### Visualiser compatibility

The following SHAP visualisers render tabular or interpretable SHAP values and
produce aggregated summaries:

- `ShapBarVisualiser`
- `ShapBeeswarmVisualiser`

The following SHAP visualisers render one selected sample and preserve local
scope:

- `ShapForceVisualiser`
- `ShapWaterfallVisualiser`

`ShapImageVisualiser` is only compatible with:

- `GradientExplainer`
- `DeepExplainer`

#### `ShapImageVisualiser` configuration

`ShapImageVisualiser` uses a **custom Matplotlib-based implementation** rather than SHAP's native `image_plot`. This provides:

- Consistent paired image/overlay layout across RAITAP visualisers
- Sample-aware titles with configurable naming
- Flexible colorbar and overlay control
- Original image panels alongside attribution heatmaps

The visualiser renders pixel-level SHAP attributions as heatmaps with positive contributions in warm colours and negative contributions in cool colours.

##### Constructor parameters

Configure these via the `constructor` key when defining the visualiser:

| Parameter                | Type          | Default      | Description                                                     |
| ------------------------ | ------------- | ------------ | --------------------------------------------------------------- |
| `max_samples`            | `int`         | `4`          | Maximum number of images to display side by side                |
| `title`                  | `str \| None` | `None`       | Optional attribution panel title (falls back to algorithm name) |
| `include_original_image` | `bool`        | `True`       | Whether to render original image next to attribution heatmap    |
| `show_colorbar`          | `bool`        | `True`       | Whether to add a SHAP colorbar in the paired layout             |
| `cmap`                   | `str`         | `"coolwarm"` | Matplotlib colormap for the SHAP heatmap overlay                |
| `overlay_alpha`          | `float`       | `0.65`       | Alpha value for the SHAP heatmap overlay                        |

Image visualisers that render an original input next to an attribution declare
`embeds_original_input = True` and accept the runtime kwarg
`include_original_input`. Reporting uses that contract to suppress repeated
original panels in the compact local layout. The older runtime kwarg
`include_original_image` is still accepted for one release with a deprecation
warning; constructor configuration keeps the existing `include_original_image`
name for YAML compatibility.

##### Call parameters

Override these via the visualiser `call` key or at runtime:

`sample_names` usually comes from the explainer's `raitap.sample_names` metadata, but
you can still override it directly on the visualiser call when needed. When set, the
list length must equal the number of input samples `N`; a mismatch raises
`raitap.utils.errors.SampleNamesLengthError` at factory entry. Omit `sample_names`
to fall back to auto-derived sample ids from the data loader.
`show_sample_names` follows the same pattern: set the shared default under
`raitap.show_sample_names` on the explainer, then override it per visualiser
via `visualisers[].call.show_sample_names` when one renderer should behave
differently.

| Parameter           | Type                | Default | Description                                                          |
| ------------------- | ------------------- | ------- | -------------------------------------------------------------------- |
| `max_samples`       | `int \| None`       | `None`  | Runtime override for maximum samples to display                      |
| `sample_names`      | `list[str] \| None` | `None`  | Optional names per sample                                            |
| `show_sample_names` | `bool`              | `False` | Whether to render sample names in subplot titles                     |
| `title`             | `str \| None`       | `None`  | Runtime override for attribution title (even empty string preserved) |
| `algorithm`         | `str \| None`       | `None`  | Explainer algorithm name (used for default title rendering)          |

##### Configuration example

```{config-tabs}
:yaml:
transparency:
  my_shap_explainer:
    _target_: "ShapExplainer"
    algorithm: "GradientExplainer"
    constructor:
      local_smoothing: 0.0
    call:
      target: 0
      background_data:
        source: imagenet_samples
        n_samples: 50
    raitap:
      batch_size: 1
    visualisers:
      # Minimal configuration
      - _target_: "ShapImageVisualiser"
        constructor:
          max_samples: 1

      # Full configuration with all options
      - _target_: "ShapImageVisualiser"
        constructor:
          max_samples: 2
          title: "Tumour attribution"
          include_original_image: true
          show_colorbar: true
          cmap: "coolwarm"
          overlay_alpha: 0.65
        call:
          show_sample_names: true

:python:
from raitap.transparency import shap, shap_image

transparency = {
    "my_shap_explainer": shap(
        algorithm="GradientExplainer",
        constructor={"local_smoothing": 0.0},
        call={
            "target": 0,
            "background_data": {
                "source": "imagenet_samples",
                "n_samples": 50,
            },
        },
        raitap={"batch_size": 1},
        visualisers=[
            # Minimal configuration.
            shap_image(max_samples=1),
            # Full configuration — builder takes flat constructor kwargs
            # directly; ``call=`` carries render-time options.
            shap_image(
                max_samples=2,
                title="Tumour attribution",
                include_original_image=True,
                show_colorbar=True,
                cmap="coolwarm",
                overlay_alpha=0.65,
                call={"show_sample_names": True},
            ),
        ],
    ),
}
```

**Note:** `ShapImageVisualiser` requires pixel-level SHAP values from `GradientExplainer` or `DeepExplainer`. Other SHAP visualisers are intended for tabular or interpretable feature outputs and are treated as aggregated summaries when they aggregate the selected batch.

## Third-party adapters

Third-party adapters published to PyPI can register under the `raitap.adapters`
entry-point group and are auto-discovered at config-registration time. Once
installed they appear alongside in-tree explainers: `+transparency=myexplainer`
in the CLI or `from raitap.transparency import myexplainer` in Python. See
{doc}`../../contributor/writing-a-plugin`.
