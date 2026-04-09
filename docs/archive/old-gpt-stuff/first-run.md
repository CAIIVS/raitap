# First Run

This tutorial walks through the shortest path to a successful local run using the
default CLI entrypoint.

## 1. Install a CPU runtime

```bash
uv sync --extra torch-cpu
```

If you want explainability backends available immediately, install the transparency
bundle instead:

```bash
uv sync --extra torch-cpu --extra transparency
```

## 2. Run the default assessment

The `raitap` console script points to `raitap.run.__main__:main`, which loads the Hydra
config tree rooted at `src/raitap/configs/`.

Run the default configuration:

```bash
uv run raitap
```

The default config currently composes:

- `transparency=demo`
- `model=vit_b32`
- `data=isic2018`
- `metrics=classification`
- `tracking=null`
- `hardware=gpu`

## 3. Inspect the resolved config

Hydra can print the full resolved job configuration without running the pipeline:

```bash
uv run raitap --cfg job
```

Use this command whenever you want to verify the exact presets and overrides that will
be applied.

## 4. Try a few focused overrides

Switch to a different model and dataset:

```bash
uv run raitap model=resnet50 data=imagenet_samples
```

Select a different transparency preset:

```bash
uv run raitap transparency=shap_gradient
```

Force CPU execution:

```bash
uv run raitap hardware=cpu
```

## 5. Find the output directory

Every run writes to Hydra's output directory. The CLI summary prints the resolved output
path at the end of the run.

The generated directory contains run metadata plus any explanation or visualisation
artifacts produced by the selected transparency configuration.

## Next steps

- Learn the override patterns in [](../how-to/hydra-overrides.md)
- Enable tracking with [](../how-to/mlflow.md)
- Review the config surface in [](../reference/config/index.md)
