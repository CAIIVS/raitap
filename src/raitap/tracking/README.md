# Smoke Test

This directory contains utility scripts for local validation.

## `smoke_test_mlflow.py`

`src/raitap/tracking/smoke_test_mlflow.py` is a small end-to-end check for the current
MLflow integration.

It verifies that RAITAP can:

- load a model
- load input data
- compute a transparency result with Captum
- compute classification metrics
- log config, dataset metadata, metrics, and transparency artifacts to MLflow

The current logging split is:

- `MetricsEvaluation.log(tracker)` (after `Metrics(...)`) owns metric-to-tracker translation
- `ExplanationResult.log` / `VisualisationResult.log` own transparency-to-tracker translation
- the tracking backend only handles generic MLflow logging calls

## What It Uses

By default the script uses:

- model: `resnet50`
- image: `~/.cache/raitap/imagenet_samples/golden_retriever.jpg`
- transparency method: `CaptumExplainer` with `IntegratedGradients`
- tracking backend: local MLflow server at `http://127.0.0.1:5000`
- backend store: local SQLite database `./mlflow/mlflow.db`
- artifact root: `./mlflow/artifacts`
- local artifacts: `./outputs/smoke-manual`

The script predicts the target class automatically from the model output and
passes it to Captum. This is necessary because gradient-based explainers for
multiclass models need a target class.

## Important Limitation

The smoke test metrics are not a real evaluation.

For the smoke test, the predicted class is also used as the target label. That
means the logged metrics are expected to be perfect:

- `performance.accuracy = 1.0`
- `performance.precision = 1.0`
- `performance.recall = 1.0`
- `performance.f1 = 1.0`

This is intentional. The point is to verify that the metrics pipeline and
MLflow logging work, not to assess model quality.

## Prerequisites

Install dependencies first:

```bash
uv sync --group dev --extra torch-cpu --extra captum --extra metrics --extra mlflow
```

The default sample image must exist locally. If it is missing, either:

- run the app once with `data=imagenet_samples`, or
- pass a local image path with `--image`

## Run The Smoke Test

From the repository root:

```bash
uv run raitap-mlflow-server
```

The launcher reads its defaults from `src/raitap/configs/tracking/mlflow_server.yaml`.

Then in a second terminal:

```bash
uv run python -m raitap.tracking.smoke_test_mlflow
```

You can also use a different image:

```bash
uv run python -m raitap.tracking.smoke_test_mlflow --image /path/to/local/image.jpg
```

If you want to log the PyTorch model artifact too:

```bash
uv run python -m raitap.tracking.smoke_test_mlflow --log-model
```

## Inspect The Results

After a successful run, local artifacts are written to:

```text
outputs/smoke-manual/
```

Current structure:

```text
outputs/smoke-manual/
  metrics/
    metrics.json
    artifacts.json
    metadata.json
  transparency/
    attributions.pt
    CaptumImageVisualiser.png
    metadata.json
```

## Open The MLflow UI

### Local MLflow server with SQLite

This is the default and recommended setup.

Start the server:

```bash
uv run raitap-mlflow-server
```

Run the smoke test against the server:

```bash
uv run python -m raitap.tracking.smoke_test_mlflow
```

Then open:

```text
http://127.0.0.1:5000
```

## What You Should See In MLflow

In the run page:

- Params:
    - assessment and model/data config
- Metrics:
    - `performance.accuracy`
    - `performance.precision`
    - `performance.recall`
    - `performance.f1`
- Artifacts:
    - `config/config.json`
    - `dataset/dataset.json`
    - `metrics/...`
    - `transparency/...`

The transparency visualizer image is expected to appear under the `Artifacts`
tab, not under `Metrics`.

## Troubleshooting

### Cannot connect to tracking server

If the smoke test cannot reach `http://127.0.0.1:5000`, start the local MLflow
server first:

```bash
uv run raitap-mlflow-server
```

### No metrics visible in MLflow

Only runs created with the current version of `smoke_test_mlflow.py` log
metrics. Re-run the script if you only see transparency artifacts.

### Captum complains about missing target

That should not happen in the smoke test script, because it computes the target
from the model prediction before calling `explain(...)`. If it happens, check
that the selected model returns logits with shape `(N, C)`.
