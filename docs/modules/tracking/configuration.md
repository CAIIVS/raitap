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

:option: backend_store_uri
:allowed: string, null
:default: "sqlite:///mlflow/mlflow.db"
:description: MLflow backend store URI used when RAITAP starts a local MLflow
  server, or when `output_forwarding_url` is not set. SQLite URIs such as
  `sqlite:///mlflow/mlflow.db` are supported for local tracking storage.

:option: default_artifact_root
:allowed: string, null
:default: "./mlflow/artifacts"
:description: MLflow artifact root used when RAITAP manages the local MLflow
  backend store or starts a local MLflow server. If `output_forwarding_url`
  points to an explicit local file store and this option is not set, MLflow's
  default artifact location is used.

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
  backend_store_uri: "sqlite:///mlflow/mlflow.db"
  default_artifact_root: "./mlflow/artifacts"
  log_model: true

:cli: +tracking=mlflow tracking.log_model=true
```
