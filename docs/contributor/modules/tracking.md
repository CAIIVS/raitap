---
title: "Contributing to the tracking module"
description: "Internal tracking architecture: BaseTracker interface, runtime call sequence, model logging, and auto-opening."
myst:
  html_meta:
    "description": "Internal tracking architecture: BaseTracker interface, runtime call sequence, model logging, and auto-opening."
---

# Contributing to the tracking module

Tracking is a backend plugin selected via the top-level `tracking.use` field. The current built-in backend is `MLFlowTracker` (`use: mlflow`).

## `BaseTracker` interface

- `log_config()`
- `log_model()`
- `log_dataset()`
- `log_artifacts()`
- `log_metrics()`
- `terminate()`

## Runtime flow

Tracking runs **after** the local assessment phase. `src/raitap/pipeline/orchestrator.py` first runs the forward pass, optional metrics, and optional transparency artifacts in the Hydra run directory. Only then, if tracking is enabled, it constructs the tracker and calls:

1. `log_config()`
2. `log_model()` if `tracking.log_model=true`
3. `log_dataset()`
4. metric scalars and metric artifacts
5. transparency explanation artifacts
6. transparency visualisation artifacts

This split keeps the metrics and transparency modules responsible for their own tracker-facing translations; the tracker backend only handles generic logging calls.

## Model logging

Enabled with `tracking.log_model=true`.

- Torch-backed models → `mlflow.pytorch.log_model(...)`.
- ONNX-backed models → `mlflow.onnx.log_model(...)`. Requires the `onnx` package in addition to `mlflow`.

## Auto-opening

If `tracking.open_when_done=true`, `terminate()` must open the tracking UI automatically. See `MLFlowTracker` in `src/raitap/tracking/mlflow_tracker.py` for the reference implementation.

## Extension points

See {doc}`../adding/adding-an-adapter`.
