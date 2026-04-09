```{config-page}
:intro: This page describes how RAITAP stores run outputs and exposes them to
  external tracking systems.

:option: _target_
:allowed: string
:default: "MLFlowTracker"
:description: Hydra target for the tracking backend.

:option: output_forwarding_url
:allowed: string, null
:default: null
:description: Optional URL where generated outputs should be forwarded.

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
  output_forwarding_url: null
  log_model: false
  open_when_done: false

:cli: tracking.log_model=true tracking.open_when_done=true
```
