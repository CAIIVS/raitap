# Config Groups

The top-level Hydra composition is defined in `src/raitap/configs/config.yaml`.

Current defaults:

- `transparency=demo`
- `model=vit_b32`
- `data=isic2018`
- `metrics=classification`
- `tracking=null`
- `hardware=gpu`
- `experiment_name=demo`

## Available config groups

### `model`

Available presets:

- `vit_b32`
- `resnet50`

These files live in `src/raitap/configs/model/`.

### `data`

Available presets:

- `isic2018`
- `imagenet_samples`
- `malaria`
- `udacityselfdriving`

These files live in `src/raitap/configs/data/`.

### `transparency`

Available presets:

- `demo`
- `onnx_demo`
- `shap_deep`
- `shap_gradient`

These files live in `src/raitap/configs/transparency/`.

The `demo` preset is currently the default and defines multiple named explainers under
the `transparency` mapping.

### `metrics`

Available presets:

- `classification`
- `detection`

These files live in `src/raitap/configs/metrics/`.

### `tracking`

Available presets:

- `null`
- `mlflow`

The MLflow-specific files live in `src/raitap/configs/tracking/`.

## Top-level schema

The structured application schema lives in `src/raitap/configs/schema.py`.

Important top-level fields:

- `model`
- `data`
- `transparency`
- `metrics`
- `tracking`
- `hardware`
- `experiment_name`

## Practical override examples

Switch a preset:

```bash
uv run raitap model=resnet50
```

Combine multiple preset selections:

```bash
uv run raitap model=resnet50 data=imagenet_samples transparency=shap_gradient
```

Override a single scalar field:

```bash
uv run raitap hardware=cpu experiment_name=demo_cpu
```
