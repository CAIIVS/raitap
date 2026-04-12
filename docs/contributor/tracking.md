# Contributing to the tracking module

This page describes the internal tracking architecture and the current MLflow
integration.

## Overview

Tracking is implemented as a backend plugin selected through the top-level
`tracking._target_` field. The current built-in backend is `MLFlowTracker`.

All trackers implement `BaseTracker`, which defines the common surface:

- `log_config()`
- `log_model()`
- `log_dataset()`
- `log_artifacts()`
- `log_metrics()`
- `terminate()`

## Runtime flow

Tracking runs after the local assessment phase. In `src/raitap/run/pipeline.py`,
RAITAP first computes the forward pass, optional metrics, and optional
transparency artifacts in the normal Hydra run directory.

If tracking is enabled, RAITAP then creates the tracker and logs data in this
order:

1. `log_config()`
2. `log_model()` if `tracking.log_model=true`
3. `log_dataset()`
4. metric scalars and metric artifacts
5. transparency explanation artifacts
6. transparency visualisation artifacts

This split keeps metrics and transparency responsible for their own
tracker-facing translations, while the tracker backend only handles generic
logging calls.

## Model logging

Model logging is optional and is enabled with `tracking.log_model=true`.

- Torch-backed models are logged through `mlflow.pytorch.log_model(...)`
- ONNX-backed models are logged through `mlflow.onnx.log_model(...)`

ONNX logging requires the `onnx` package in addition to `mlflow`.

## Auto-opening

If `tracking.open_when_done=true`, the tracking UI must be opened automatically when `terminate()` is called. See how `MLFlowTracker` does this in `src/raitap/tracking/mlflow_tracker.py`.

## Extension points

To add another tracking backend:

1. Implement `BaseTracker`
2. Export the class from `raitap.tracking`
3. Add a config preset under `src/raitap/configs/tracking/`
4. Update the user-facing tracking docs if the new backend changes available
   behavior
