# Runtime Pipeline

The main assessment flow lives in `src/raitap/run/pipeline.py`.

## High-level sequence

```text
AppConfig
  -> Model(config)
  -> Data(config)
  -> forward pass
  -> optional metrics
  -> explanations
  -> visualisations
  -> optional tracking
  -> RunOutputs
```

## Step-by-step behavior

1. `Model(config)` resolves the runtime backend and loads the selected model.
2. `Data(config)` loads the configured dataset or sample source.
3. The pipeline prints a short summary of the selected experiment, dataset, hardware, and
   output directory.
4. The model backend prepares the input tensor and runs a forward pass.
5. If metrics are enabled, the pipeline computes predictions and resolves metric targets.
6. The pipeline iterates over every configured transparency entry and creates an
   explanation result for each named explainer.
7. Each explanation result renders one or more visualisations.
8. If tracking is enabled, the tracker logs config, data, optional model artifacts,
   optional metrics, explanations, and visualisations.
9. The pipeline returns a `RunOutputs` object containing the forward output and generated
   artifacts.

## Why this matters for documentation

This design means the user-facing documentation must explain both:

- the CLI and Hydra composition layer
- the runtime data flow through model loading, explanation generation, metrics, and
  tracking

It is not enough to document classes in isolation, because the main behavior emerges from
how `AppConfig` drives the pipeline.
