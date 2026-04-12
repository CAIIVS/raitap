# Supported tracking backends

The tracking module logs experiment metadata, metrics, and artifacts to external backends.

## Tracking workflow

RAITAP writes outputs to the local directory first, then forwards them to the tracking backend. The local output directory remains intact regardless of the tracking configuration.

## MLflow

### Docs

[MLflow docs](https://mlflow.org/docs/latest/ml/)

### Configuration

```yaml
tracking:
  _target_: MLFlowTracker
  output_forwarding_url: http://127.0.0.1:5000
  log_model: true
  open_when_done: false
```

If `output_forwarding_url` is not set, MLflow stores runs locally in `./mlruns`.

### Logged artifacts

For each run, MLflow receives:

- Configuration as JSON
- Dataset metadata
- Scalar performance metrics
- Transparency outputs (attributions and visualizations)
- Model artifacts (when `log_model: true`)

### Model logging

When `log_model: true`, RAITAP logs the assessed model to MLflow. This can take significant time and resources for large models.
