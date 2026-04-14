# Supported libraries

## `constructor` and `call` keys

Both explainers and visualisers support the `constructor` and `call` keys. They pass `kwargs` to the constructor and to the runtime method (`explain` or `visualise`).

This allows you to configure the underlying library object. Here an example:

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

Override these via the `call` key or at runtime:

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

(alibi-frameworks)=

### Alibi

#### Docs

- [Alibi Explain](https://docs.seldon.io/projects/alibi/en/stable/overview/high_level.html)

#### Installation and license

:::{warning}
**Alibi Explain is under Seldon’s Business Source License 1.1 (BSL 1.1)** — not GPLv3. Non-production use is permitted on Seldon’s terms; production or commercial use may require a separate license. RAITAP (GPLv3) does not relicense Alibi. Read [Seldon’s license](https://github.com/SeldonIO/alibi/blob/master/LICENSE) before using.
:::

`alibi` 0.9.x hard-pins three packages that conflict with RAITAP’s own requirements:

| alibi pins          | RAITAP requires                   |
| ------------------- | --------------------------------- |
| `numpy<2.0.0`       | `numpy>=2.4.2`                    |
| `Pillow<11.0`       | `pillow>=10.0.0` (currently 12.x) |
| `scikit-image<0.23` | (currently 0.26)                  |

The `alibi` extra is provided, but you must add the following **overrides to your own `pyproject.toml`** so uv ignores alibi’s upper bounds:

```toml
[tool.uv.override-dependencies]
numpy = ["numpy>=2.4"]
Pillow = ["Pillow>=12.0"]
scikit-image = ["scikit-image>=0.26"]
```

Then install normally:

```sh
uv add "raitap[alibi]"
```

:::{note}
These overrides bypass version constraints declared by alibi but do not guarantee alibi works correctly with those newer versions — Seldon has not tested or supported this combination. RAITAP’s `KernelShap` path has been validated; other algorithms may behave differently. The `alibi` extra will be cleaned up once Seldon ships a NumPy 2-compatible release.
:::

#### Explainers

`AlibiExplainer` wraps a subset of [Alibi explainers](https://docs.seldon.io/projects/alibi/en/stable/api/alibi.explainers.html):

- **`KernelShap`** — black-box SHAP-style explanations. RAITAP passes a NumPy batch through your **`torch.nn.Module`** (converted to tensors on the model’s device). Optional `call` keys include `background_data`, `task` (`"classification"` / `"regression"`), `nsamples`, and `target` (class index for classification).
- **`IntegratedGradients`** — Alibi’s TensorFlow/Keras API only. Put a **`keras_model`** (`tf.keras.Model`) in the Hydra **`constructor`** block. For PyTorch integrated gradients, use **Captum** or **`KernelShap`** here.

Example (tabular-oriented preset lives under `src/raitap/configs/transparency/alibi_kernel.yaml`):

```yaml
transparency:
  my_alibi_explainer:
    _target_: AlibiExplainer
    algorithm: KernelShap
    call:
      nsamples: 32
      task: classification
    visualisers:
      - _target_: TabularBarChartVisualiser
```

#### ONNX compatibility

RAITAP does not expose an ONNXRuntime-specific Alibi path. **`KernelShap` requires a PyTorch `nn.Module`** whose `forward` is invoked on tensor batches (Alibi calls your model from NumPy inputs). If your deployment uses ONNX only, use another explainer or wrap inference in an `nn.Module` that matches this contract.

#### Visualiser compatibility

`AlibiExplainer` produces the same kind of **tensor attributions** as Captum/SHAP (heat-map compatible), so pick RAITAP visualisers that match your input modality (for example **`TabularBarChartVisualiser`** for flat/tabular features). The sample config pairs `KernelShap` with `TabularBarChartVisualiser`; image or text pipelines should use the corresponding RAITAP image/text visualisers where shapes align.
