---
title: "Supported tracking backends"
description: "The tracking module logs experiment metadata, metrics, and artifacts to external backends."
myst:
  html_meta:
    "description": "The tracking module logs experiment metadata, metrics, and artifacts to external backends."
---

# Supported tracking backends

The tracking module logs experiment metadata, metrics, and artifacts to external backends.

## Tracking workflow

RAITAP writes outputs to the local directory first, then forwards them to the tracking backend. The local output directory remains intact regardless of the tracking configuration.

## Stopping detached tracker processes

Some trackers start background servers or UIs that outlive a single run. Shut
them down with:

```bash
uv run raitap tracking stop
```

## MLflow

### Docs

[MLflow docs](https://mlflow.org/docs/latest/ml/)

### Configuration

```{config-tabs}
:yaml:
tracking:
  use: mlflow
  output_forwarding_url: http://127.0.0.1:5001
  log_model: true
  open_when_done: false

:python:
from raitap.tracking import mlflow

tracking = mlflow(
    output_forwarding_url="http://127.0.0.1:5001",
    log_model=True,
    open_when_done=False,
)
```

If `output_forwarding_url` is not set, MLflow uses
`sqlite:///mlflow/mlflow.db`. From the repository root, the database is stored
at `mlflow/mlflow.db` and artifacts are stored under `mlflow/artifacts`.
Users can still point `output_forwarding_url` at an existing HTTP tracking
server.

To migrate an existing file-store run history:

```bash
uv run mlflow migrate-filestore --source ./mlruns --target sqlite:///mlflow/mlflow.db
```

### Logged artifacts

For each run, MLflow receives:

- Configuration as JSON
- Dataset metadata
- Scalar performance metrics
- Transparency outputs (attributions and visualisations)
- Model artifacts (when `log_model: true`)

### Model logging

When `log_model: true`, RAITAP logs the assessed model to MLflow. This can take significant time and resources for large models.

## Third-party adapters

Third-party adapters published to PyPI can register under the `raitap.adapters`
entry-point group and are auto-discovered at config-registration time. Once
installed they appear alongside in-tree trackers. See
{doc}`../../contributor/writing-a-plugin`.
