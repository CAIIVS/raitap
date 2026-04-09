# Frameworks and libraries

## Core libraries

The current transparency module relies on:

- [`captum`](https://captum.ai/) for gradient-based and perturbation-based attribution methods
- [`shap`](https://shap.readthedocs.io/en/latest/) for Shapley-style explainers
- [`matplotlib`](https://matplotlib.org/) for writing visualisations to PNG artifacts

## Captum explainers

`CaptumExplainer` wraps classes under `captum.attr`, such as
`IntegratedGradients`, `Saliency`, or `LayerGradCam`.

### Captum configuration model

Captum uses the standard RAITAP transparency config shape:

- `constructor` contains arguments for the Captum method constructor
- `call` contains arguments passed when computing attributions

Example:

```yaml
transparency:
  captum_ig:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    constructor: {}
    call:
      target: 0
```

### Captum-specific notes

- `target` and `baselines` are typical `call` arguments
- `LayerGradCam` also accepts `constructor.layer_path`, which RAITAP resolves on
  the model before instantiating the Captum method

### Captum backend support

For ONNX-backed models, RAITAP restricts Captum to algorithms that do not
depend on Torch autograd:

- `FeatureAblation`
- `FeaturePermutation`
- `Occlusion`
- `ShapleyValueSampling`
- `ShapleyValues`
- `KernelShap`
- `Lime`

## SHAP explainers

`ShapExplainer` wraps classes available in `shap`, such as
`GradientExplainer`, `DeepExplainer`, or `KernelExplainer`.

### SHAP configuration model

SHAP uses the same RAITAP transparency config shape:

- `constructor` contains arguments for the SHAP explainer constructor
- `call` contains arguments passed when computing attributions

Example:

```yaml
transparency:
  shap_gradient:
    _target_: ShapExplainer
    algorithm: GradientExplainer
    constructor: {}
    call:
      target: 0
      background_data:
        source: imagenet_samples
```

### SHAP-specific notes

- `background_data` is usually passed under `call` and can be loaded from a
  runtime data source
- execution-time arguments are forwarded to `shap_values(...)`

For `GradientExplainer`, `DeepExplainer`, and `KernelExplainer`, SHAP normally
expects background data. If none is provided, RAITAP falls back to the input
batch.

### SHAP backend support

For ONNX-backed models, RAITAP currently supports only `KernelExplainer`.

## Visualisers

Visualisers are configured separately from explainers. The current public
surface includes:

- Captum visualisers: `CaptumImageVisualiser`, `CaptumTextVisualiser`,
  `CaptumTimeSeriesVisualiser`
- SHAP visualisers: `ShapBarVisualiser`, `ShapBeeswarmVisualiser`,
  `ShapForceVisualiser`, `ShapImageVisualiser`, `ShapWaterfallVisualiser`
- framework-agnostic visualisers: `TabularBarChartVisualiser`

RAITAP will warn you if you attempt to use a visualiser with an incompatible algorithm.
