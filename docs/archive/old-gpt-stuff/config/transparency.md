# Transparency Configuration

The transparency subsystem is driven by a mapping of named explainers. Each entry is a
small Hydra object specification that resolves an explainer plus one or more visualisers.

## Config shape

Each named transparency entry follows this shape:

```yaml
transparency:
  my_explainer:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    constructor: {}
    call: {}
    visualisers:
      - _target_: CaptumImageVisualiser
        constructor: {}
        call: {}
```

## Field meanings

- `_target_`: explainer class name or fully-qualified import path
- `algorithm`: backend-specific algorithm name, such as `IntegratedGradients` or
  `GradientExplainer`
- `constructor`: keyword arguments passed into the explainer constructor
- `call`: keyword arguments passed each time the explainer computes attributions
- `visualisers`: a list of visualiser specifications, each with its own `_target_`,
  optional `constructor`, and optional `call`

## `constructor` versus `call`

RAITAP validates explainer and visualiser configs separately. Use:

- `constructor` for `__init__` arguments
- `call` for execution-time parameters

For example, `local_smoothing` belongs in the SHAP explainer constructor:

```yaml
shap_gradient:
  _target_: ShapExplainer
  algorithm: GradientExplainer
  constructor:
    local_smoothing: 0.0
```

The target class and batch parameters belong in `call`:

```yaml
shap_gradient:
  _target_: ShapExplainer
  algorithm: GradientExplainer
  call:
    target: 0
    nsamples: 10
    batch_size: 1
```

## Data-source resolution inside `call`

Any `call` value shaped like a plain mapping with a `source` key is interpreted as a
data-source reference and loaded into a tensor at runtime.

Example:

```yaml
shap_gradient:
  _target_: ShapExplainer
  algorithm: GradientExplainer
  call:
    background_data:
      source: imagenet_samples
      n_samples: 50
```

Supported fields for this pattern:

- `source` (required)
- `n_samples` (optional)

## Visualiser compatibility

Visualisers declare compatible algorithms, and RAITAP validates the chosen algorithm
before it attempts a run. If the algorithm and visualiser do not match, the pipeline
raises a compatibility error rather than failing late.

## Default transparency presets

The repository currently ships these preset files:

- `src/raitap/configs/transparency/demo.yaml`
- `src/raitap/configs/transparency/onnx_demo.yaml`
- `src/raitap/configs/transparency/shap_deep.yaml`
- `src/raitap/configs/transparency/shap_gradient.yaml`

## CLI examples

Select a preset:

```bash
uv run raitap transparency=shap_gradient
```

Override the algorithm inside a named explainer entry:

```bash
uv run raitap transparency.captum_ig.algorithm=GradientShap
```

Replace the visualiser list:

```bash
uv run raitap "transparency.captum_ig.visualisers=[{_target_: CaptumImageVisualiser}]"
```
