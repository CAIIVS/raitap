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
