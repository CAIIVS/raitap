---
title: "Contributing to the metrics module"
description: "Internal metrics architecture: BaseMetricComputer interface, MetricResult shape, and output file layout."
myst:
  html_meta:
    "description": "Internal metrics architecture: BaseMetricComputer interface, MetricResult shape, and output file layout."
---

# Contributing to the metrics module

Metric computers wrap evaluation libraries (torchmetrics, faster-coco-eval, ...) behind a unified interface driven by Hydra `_target_` instantiation.

## `BaseMetricComputer` interface

All metric computers implement three methods:

- `reset() -> None` — clear internal state.
- `update(predictions, targets) -> None` — accumulate a batch.
- `compute() -> MetricResult` — finalise and return results.

## `MetricResult` shape

```python
@dataclass
class MetricResult:
    metrics: dict[str, float]      # scalar values → metrics.json
    artifacts: dict[str, Any]      # structured outputs → artifacts.json
```

## Output file layout

The metrics module writes three files under the Hydra run folder:

```text
metrics/
├── metrics.json      # scalar results
├── artifacts.json    # structured outputs (e.g., per-class values)
└── metadata.json     # config and experiment metadata
```

## Important files

- `src/raitap/metrics/base_metric_computer.py` — `BaseMetricComputer` and `MetricResult`.
- `src/raitap/metrics/factory.py` — `evaluate()` entry point; uses Hydra `instantiate()` and resolves bare class names to `raitap.metrics.*`.
- `src/raitap/metrics/classification_metrics.py`, `detection_metrics.py` — reference implementations.

## Runtime flow

`src/raitap/metrics/phase.py` (`evaluate_metrics`) runs after the forward pass, instantiates the configured metric computer via `src/raitap/metrics/factory.py::evaluate`, calls `update(predictions, targets)` per batch, then calls `compute()` and writes the three output files.

## Extension points

The base class handles device management through `_prepare_inputs(predictions, targets)`, which moves tensors to the appropriate device before `update()` sees them. Override it only when your library needs a different input layout. For metrics that maintain internal state (torchmetrics classes etc.), store the underlying object as an instance attribute, forward to its `update()` from yours, and translate its `compute()` output into the `MetricResult` structure.

## Adding a new metric computer

See {doc}`adding-an-adapter`.
