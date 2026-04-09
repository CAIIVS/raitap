```{config-page}
:intro: This page describes how RAITAP configures an optional tracking backend.

  Tracking is disabled by default in the main application config. When enabled,
  the top-level `tracking` section selects the backend implementation and its
  runtime options.

  See [Frameworks and libraries](frameworks-and-libraries.md) for the current
  built-in backend. See [Output](output.md) for what RAITAP logs to that
  backend.

:option: _target_
:allowed: string
:default: "MLFlowTracker"
:description: Hydra target for the tracking backend implementation.

:option: output_forwarding_url
:allowed: string, null
:default: null
:description: Tracking URI used by the backend. For `MLFlowTracker`, this is
  passed to MLflow as the tracking URI. It can point to a local path or an
  HTTP endpoint.

:option: log_model
:allowed: true, false
:default: false
:description: Whether to log the assessed model to the tracking backend.

:option: open_when_done
:allowed: true, false
:default: false
:description: Whether to open the tracking UI automatically after the run
  completes.

:yaml:
tracking:
  _target_: "MLFlowTracker"
  output_forwarding_url: "http://127.0.0.1:5000"
  log_model: false
  open_when_done: true

:cli: tracking=mlflow tracking.log_model=true
```
