# Supported libraries

## `constructor`, `call`, and `raitap` keys

Explainers support three config buckets:

- `constructor`: kwargs for the explainer constructor or underlying library object
- `call`: verbatim library kwargs for the underlying attribution call
- `raitap`: RAITAP-owned runtime options such as batching, progress bars, and sample-name metadata

Visualisers continue to support `constructor` and `call` only.

This keeps the boundary clear for users: `call` is what Captum, SHAP, or Alibi sees, while `raitap` is what RAITAP itself consumes. Example:

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

(alibi-frameworks)=

### Alibi

#### Docs

- [Alibi Explain](https://docs.seldon.io/projects/alibi/en/stable/overview/high_level.html)

(alibi-install-overrides)=

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

Independently, Alibi’s **spaCy** dependency can resolve to **thinc** / **blis** versions that lack **Python 3.13** wheels and fall back to **sdist** builds (often failing). The `alibi` extra is supported, but you must add **`[tool.uv]` overrides** in your own **`pyproject.toml`** — use the same entries as RAITAP’s `pyproject.toml`:

```toml
[tool.uv]
override-dependencies = [
  "numpy>=2.4",
  "Pillow>=12.0",
  "scikit-image>=0.26",
  "blis>=1.0.2",
  "thinc>=8.3.6,<9",
  "spacy>=3.8.0",
]
```

Then install with **uv** (recommended):

```sh
uv add "raitap[alibi]"
```

:::{note}
**pip** (and other installers that are not **uv**) do **not** read `override-dependencies`. Prefer **uv** for `raitap[alibi]`; if you use **pip**, you must satisfy compatible versions of the packages above yourself — RAITAP does not document a supported **pip-only** recipe.
:::

:::{note}
These overrides bypass version constraints declared by Alibi and its transitive dependencies but do not guarantee every Alibi algorithm works with those newer versions — Seldon has not tested or supported this combination. RAITAP’s **`KernelShap`** path is exercised in tests (including under **SHAP 0.5x**, where RAITAP adapts stacked multi-class outputs before Alibi builds explanation metadata); other algorithms may behave differently. The `alibi` extra will be cleaned up once upstream metadata and wheels align with RAITAP’s baseline.
:::

#### Explainers

`AlibiExplainer` wraps a subset of [Alibi explainers](https://docs.seldon.io/projects/alibi/en/stable/api/alibi.explainers.html):

- **`KernelShap`** — black-box SHAP-style explanations. RAITAP passes a NumPy batch through your **`torch.nn.Module`** (converted to tensors on the model’s device). Optional `call` keys include `background_data`, `task` (`"classification"` / `"regression"`), `nsamples`, and `target` (class index for classification).
- **`IntegratedGradients`** — Alibi’s TensorFlow/Keras API only. Put a **`keras_model`** (`tf.keras.Model`) in the Hydra **`constructor`** block. For PyTorch integrated gradients, use **Captum** or **`KernelShap`** here.

RAITAP explainer-level metadata keys `raitap.sample_names` and `raitap.show_sample_names`
are honoured for downstream visualisers as the default metadata values. If a
specific visualiser needs different sample-name rendering, override it under
`visualisers[].call.sample_names` or `visualisers[].call.show_sample_names`.
RAITAP batching/progress keys
(`raitap.batch_size`, `raitap.show_progress`, `raitap.progress_desc`) are currently
ignored for Alibi and emit a warning when set.

Example (tabular-oriented preset lives under `src/raitap/configs/transparency/alibi_kernel.yaml`):

```yaml
transparency:
  my_alibi_explainer:
    _target_: AlibiExplainer
    algorithm: KernelShap
    call:
      nsamples: 32
      task: classification
    raitap:
      show_sample_names: false
    visualisers:
      - _target_: TabularBarChartVisualiser
```

#### ONNX compatibility

RAITAP does not expose an ONNXRuntime-specific Alibi path. **`KernelShap` requires a PyTorch `nn.Module`** whose `forward` is invoked on tensor batches (Alibi calls your model from NumPy inputs). If your deployment uses ONNX only, use another explainer or wrap inference in an `nn.Module` that matches this contract.

#### Visualiser compatibility

`AlibiExplainer` produces the same kind of **tensor attributions** as Captum/SHAP (heat-map compatible), so pick RAITAP visualisers that match your input modality (for example **`TabularBarChartVisualiser`** for flat/tabular features). The sample config pairs `KernelShap` with `TabularBarChartVisualiser`; image or text pipelines should use the corresponding RAITAP image/text visualisers where shapes align.
