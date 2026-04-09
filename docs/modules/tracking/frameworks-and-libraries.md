# Frameworks and libraries

## Core libraries

The current tracking module relies on:

- [`mlflow`](https://mlflow.org/) for experiment tracking, metric logging, and artifact storage

## Tracking backends

Tracking is configured through the top-level `tracking` section. RAITAP
currently ships one built-in backend: `MLFlowTracker`, usually selected with
`tracking=mlflow`.

## MLflow backend

`MLFlowTracker` configures MLflow with `output_forwarding_url` as the tracking
URI. If that option is not set, it falls back to `./mlruns`.

For a typical run, MLflow receives:

- the resolved config
- dataset metadata
- scalar metrics
- forwarded artifacts from modules such as metrics and transparency

## Model logging

If `tracking.log_model=true`, RAITAP also logs the assessed model:

- Torch-backed models are logged through `mlflow.pytorch.log_model(...)`
- ONNX-backed models are logged through `mlflow.onnx.log_model(...)`

ONNX model logging requires the `onnx` package in addition to `mlflow`.

## Practical behavior

Tracking does not replace the local output directory. It adds an external
record of the same run in the selected backend.

For implementation details about the tracker interface, runtime order, and
backend internals, see the contributor documentation.
