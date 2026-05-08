```{config-page}
:intro: This page describes how to configure the tracking module that forwards the pipeline's output to a tracking backend.

:option: _target_
:allowed: "MLFlowTracker"
:default: "MLFlowTracker"
:description: Hydra target for the tracking backend implementation.

:option: output_forwarding_url
:allowed: string, null
:default: null
:description: Tracking URI used by the backend. For `MLFlowTracker`, this is
  passed to MLflow as the tracking URI. It can point to a local path or an
  existing HTTP tracking server. If not set, it uses
  `sqlite:///mlflow/mlflow.db`, with the database at `mlflow/mlflow.db` from
  the repository root and artifacts under `mlflow/artifacts`.

:option: log_model
:allowed: boolean
:default: false
:description: Whether to log the assessed model to the tracking backend. Note that this might take significant time and resources for large models.

:option: open_when_done
:allowed: boolean
:default: true
:description: Whether to open the tracking UI automatically after the run
  completes.

:yaml:
tracking:
  _target_: "MLFlowTracker"
  output_forwarding_url: "http://127.0.0.1:5000"
  log_model: true

:cli: +tracking=mlflow tracking.log_model=true
```
