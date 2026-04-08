# Enable MLflow Tracking

RAITAP keeps tracking disabled by default. To enable MLflow, install the optional
dependency and select the `tracking=mlflow` preset.

## 1. Install the MLflow extra

```bash
uv sync --extra torch-cpu --extra mlflow
```

If you also want explainability backends available, combine the extras:

```bash
uv sync --extra torch-cpu --extra transparency --extra mlflow
```

## 2. Start the bundled MLflow server

Run the bundled local server in one terminal:

```bash
uv run raitap-mlflow-server
```

The default server config lives in `src/raitap/configs/tracking/mlflow_server.yaml`.

## 3. Run RAITAP with tracking enabled

In a second terminal, enable the tracking preset:

```bash
uv run raitap tracking=mlflow
```

The default `tracking=mlflow` preset currently sets:

- `_target_: MLFlowTracker`
- `output_forwarding_url: http://127.0.0.1:5000`
- `log_model: false`
- `open_when_done: true`

## 4. Override the tracking target

Point RAITAP at an existing MLflow instance:

```bash
uv run raitap tracking=mlflow tracking.output_forwarding_url=http://127.0.0.1:5000
```

Enable model logging explicitly:

```bash
uv run raitap tracking=mlflow tracking.log_model=true
```

Keep the browser from opening automatically:

```bash
uv run raitap tracking=mlflow tracking.open_when_done=false
```

## 5. What gets logged

When tracking is enabled, the pipeline logs:

- the resolved config snapshot
- data metadata
- optional model artifacts when `tracking.log_model=true`
- metric results when metrics are enabled
- explanation artifacts
- visualisation artifacts

For a fuller explanation of the runtime flow, see [](../explanation/pipeline.md).
