# Contributing to the metrics module

This page describes the internal metrics architecture and how to extend it with new metric computers.

## Overview

The metrics module evaluates model predictions against ground-truth targets. Metric computers wrap underlying libraries (torchmetrics, faster-coco-eval) behind a unified interface driven by Hydra `_target_` instantiation.

All metric computers implement `BaseMetricComputer`, which defines:

- `reset() -> None`
- `update(predictions, targets) -> None`
- `compute() -> MetricResult`

The `MetricResult` dataclass contains:

- `metrics: dict[str, float]` — scalar metrics written to `metrics.json`
- `artifacts: dict[str, Any]` — structured outputs written to `artifacts.json`

## Important files

The `factory.py` module provides the `evaluate()` entry point, which uses Hydra's `instantiate()` to build metric computers from `_target_` keys. Bare class names are automatically resolved to `raitap.metrics.*` paths.

## Runtime flow

Metrics run after the forward pass in `src/raitap/run/pipeline.py`. RAITAP instantiates the configured metric computer, calls `update(predictions, targets)`, then calls `compute()` to generate the final results.

The metrics module writes three files under the Hydra run folder:

```text
metrics/
├── metrics.json      # scalar results
├── artifacts.json    # structured outputs (e.g., per-class values)
└── metadata.json     # config and experiment metadata
```

## Adding a new metric computer

To add a new metric type:

1. **Implement the metric computer**

    Create a new metric computer class that extends `BaseMetricComputer` in `src/raitap/metrics/`:

    ```python
    # src/raitap/metrics/new_metrics.py
    from .base_metric import BaseMetricComputer, MetricResult
    import torch

    class NewMetrics(BaseMetricComputer):
        def __init__(self, **config_kwargs):
            # Store configuration
            pass

        def reset(self) -> None:
            """Required: Reset internal state."""
            pass

        def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
            """Required: Update internal state with a batch of predictions and targets."""
            predictions, targets = self._prepare_inputs(predictions, targets)
            # Your update logic here
            pass

        def compute(self) -> MetricResult:
            """Required: Compute final metrics and return MetricResult."""
            return MetricResult(
                metrics={"metric_name": 0.0},  # Scalar metrics
                artifacts={}  # Structured outputs
            )
    ```

    Reference `src/raitap/metrics/classification_metrics.py` or `detection_metrics.py` for complete examples.

2. **Export from `__init__.py`**

    Export the class from the metrics package:

    ```python
    # src/raitap/metrics/__init__.py
    from .new_metrics import NewMetrics

    __all__ = [..., "NewMetrics"]
    ```

3. **Create a config preset**

    Add a config file under `src/raitap/configs/metrics/`:

    ```yaml
    # src/raitap/configs/metrics/new_metrics.yaml
    _target_: NewMetrics
    # Add configuration parameters here
    ```

4. **Use it**

    ```bash
    uv run raitap metrics=new_metrics
    uv run raitap metrics=new_metrics metrics.some_param=value
    ```

5. **Add tests**

    Create unit tests in `src/raitap/metrics/tests/` following the patterns in `test_classification_metrics.py` or `test_detection_metrics.py`.

6. **Update documentation**

    Add the new metric type to `docs/modules/metrics/frameworks-and-libraries.md` with usage examples and parameter descriptions.

## Extension points

Metric computers can wrap any evaluation library. The base class handles device management through `_prepare_inputs()`, which moves predictions and targets to the appropriate device before calling `update()`.

For metrics that maintain internal state (like torchmetrics classes), store them as instance attributes and update them in `update()`. Call their `compute()` method in your `compute()` implementation and map results to the `MetricResult` structure.
