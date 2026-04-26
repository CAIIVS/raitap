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

All three are compatible with all Captum algorithms in RAITAP.

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

The following SHAP visualisers are compatible with all SHAP algorithms:

- `ShapBarVisualiser`
- `ShapBeeswarmVisualiser`
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

**Note:** `ShapImageVisualiser` requires pixel-level SHAP values from `GradientExplainer` or `DeepExplainer`. Using it with other SHAP explainers will produce meaningless plots.
