---
title: "Tracking API"
description: "Reference for the tracker interface and the built-in MLflow implementation."
---

Tracking lives under `src/raitap/tracking/` and is intentionally separated into a generic interface and one concrete implementation.

## Imports

```python
from raitap.tracking import BaseTracker, MLFlowTracker
```

## `BaseTracker`

Source: `src/raitap/tracking/base_tracker.py`

```python
class BaseTracker(ABC):
    @staticmethod
    def create_tracker(config: AppConfig) -> BaseTracker
    def __enter__(self) -> BaseTracker
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool
    def log_config(self) -> None
    def log_model(self, model: ModelBackend | nn.Module) -> None
    def log_dataset(self, description: dict[str, Any]) -> None
    def log_artifacts(
        self,
        source_directory: str | Path | None,
        target_subdirectory: str | None = None,
    ) -> None
    def log_metrics(
        self,
        metrics: dict[str, float],
        prefix: str = "performance",
    ) -> None
    def terminate(self, successfully: bool = True) -> None
```

`create_tracker()` resolves `_target_` against the `raitap.tracking.` prefix and instantiates a partial Hydra target with the active config.

## `MLFlowTracker`

Source: `src/raitap/tracking/mlflow_tracker.py`

```python
class MLFlowTracker(BaseTracker):
    def __init__(self, config: AppConfig) -> None
    def terminate(self, successfully: bool = True) -> None
    def log_config(self) -> None
    def log_dataset(self, description: dict[str, Any]) -> None
    def log_artifacts(
        self,
        source_directory: str | Path | None,
        target_subdirectory: str | None = None,
    ) -> None
    def log_metrics(
        self,
        metrics: dict[str, float],
        prefix: str = "performance",
    ) -> None
    def log_model(self, model: Any, artifact_path: str = "model") -> None
```

Constructor behavior:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AppConfig` | — | Active config used to resolve output directories and tracking settings. |

Key behaviors:

- starts an MLflow run in `__init__`
- logs summary params derived from the resolved config
- can auto-start a local MLflow server or UI process when using localhost
- supports both `TorchBackend` and `OnnxBackend` model logging

## Common pattern

```python
from raitap.tracking import BaseTracker

with BaseTracker.create_tracker(cfg) as tracker:
    tracker.log_config()
    data.log(tracker)
```

The tracker API is purposely minimal because artifact-owning objects such as `MetricsEvaluation` and `ExplanationResult` are responsible for translating themselves into tracking operations.
