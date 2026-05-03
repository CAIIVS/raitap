# Supported libraries

## `constructor`, `call`, and `raitap` keys

Explainers support three config buckets:

- `constructor`: kwargs for the explainer constructor or underlying library object
- `call`: verbatim library kwargs for the underlying attribution call
- `raitap`: RAITAP-owned runtime options such as batching, progress bars, and sample-name metadata

Visualisers continue to support `constructor` and `call` only.

This keeps the boundary clear for users: `call` is what Captum or SHAP sees, while `raitap` is what RAITAP itself consumes. Example:

```yaml
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
```

## Typed semantics and visualiser compatibility

RAITAP uses typed scope, method-family, and output-space semantics to validate
visualisers against the explanation artifact they receive. In short:

- explainers produce typed `ExplanationResult.semantics`
- visualisers declare the payload kinds, scopes, output spaces, and method
  families they can render
- reporting places rendered figures by `VisualisationResult.scope`

See {doc}`explanation-scope` for the user-facing scope model.

| Visualiser | Consumes | Produces | Notes |
| --- | --- | --- | --- |
| `CaptumImageVisualiser` | Local image input-feature attributions and supported image/CAM spatial maps | Local visualisation | Rejects tabular, token, and time-series layouts. |
| `CaptumTextVisualiser` | Local token-sequence attributions | Local visualisation | Requires explicit token metadata. |
| `CaptumTimeSeriesVisualiser` | Local time-series attributions | Local visualisation | Requires explicit time-series metadata. |
| `ShapImageVisualiser` | Local image-shaped SHAP values from `GradientExplainer` or `DeepExplainer` | Local visualisation | Intended for pixel-level SHAP values, not tabular SHAP outputs. |
| `ShapBarVisualiser` | Local tabular or interpretable SHAP attributions | Cohort visual summary | Summarizes the selected batch or cohort, so it is reported under Cohort Explanations. |
| `ShapBeeswarmVisualiser` | Local tabular or interpretable SHAP attributions | Cohort visual summary | Summarizes attribution distributions for the selected batch or cohort. |
| `ShapForceVisualiser` | Local tabular or interpretable SHAP attributions for one selected sample | Local visualisation | Preserves local scope. |
| `ShapWaterfallVisualiser` | Local tabular or interpretable SHAP attributions for one selected sample | Local visualisation | Preserves local scope. |
| `TabularBarChartVisualiser` | Local tabular or interpretable attributions | Cohort visual summary | Uses mean absolute attribution-style aggregation for the selected batch or cohort. |

## Explainer libraries

### Captum

#### Docs

- [Explainers](https://captum.ai/api/)
- [Visualisers](https://captum.ai/api/utilities.html#visualization)

#### Explainers

`CaptumExplainer` gives access to [all Captum explainers](https://captum.ai/api/).

```yaml
transparency:
  my_captum_explainer:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    constructor: {}
    call:
      target: 0
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

```yaml
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
```

`GradientExplainer`, `DeepExplainer`, and `KernelExplainer` usually require
`background_data`. If it is not provided, RAITAP falls back to the input batch.

`DeepExplainer` can fail on PyTorch models that use `SiLU` activations (for example EfficientNet variants) due to autograd/in-place limitations. In those cases, use `GradientExplainer`.

#### ONNX compatibility

Only `KernelExplainer` is compatible.

#### Visualiser compatibility

The following SHAP visualisers render tabular or interpretable SHAP values and
produce cohort summaries:

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

##### Call parameters

Override these via the visualiser `call` key or at runtime:

`sample_names` usually comes from the explainer's `raitap.sample_names` metadata, but
you can still override it directly on the visualiser call when needed.
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

```yaml
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
```

**Note:** `ShapImageVisualiser` requires pixel-level SHAP values from `GradientExplainer` or `DeepExplainer`. Other SHAP visualisers are intended for tabular or interpretable feature outputs and are treated as cohort summaries when they aggregate the selected batch.
