---
title: "Run API"
description: "Reference for the top-level run entry points and output types re-exported from raitap.run."
---

The `raitap.run` module is the main public entry point for programmatic execution. It is defined in `src/raitap/run/__init__.py` and wraps the actual pipeline implementation in `src/raitap/run/pipeline.py`.

Source files:

- `src/raitap/run/__init__.py`
- `src/raitap/run/pipeline.py`
- `src/raitap/run/outputs.py`
- `src/raitap/run/forward_output.py`

## Imports

```python
from raitap.run import (
    PredictionSummary,
    RunOutputs,
    extract_primary_tensor,
    main,
    metrics_prediction_pair,
    print_summary,
    resolve_metric_targets,
    run,
)
```

## `run`

```python
def run(config: AppConfig) -> RunOutputs
```

Executes the full assessment pipeline: model loading, data loading, forward pass, metrics, transparency, reporting, and optional tracking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AppConfig` | — | Fully resolved Hydra config. |

Return type: `RunOutputs`

Example:

```python
from hydra import compose, initialize_config_dir
from pathlib import Path
from raitap.run import run

with initialize_config_dir(version_base="1.3", config_dir=str(Path("src/raitap/configs").resolve())):
    cfg = compose(config_name="config" overrides=["hardware=cpu", "reporting=disabled"])
    outputs = run(cfg)
```

## `print_summary`

```python
def print_summary(config: AppConfig, model: Model) -> None
```

Logs the current assessment summary, including experiment name, model source, dataset name, resolved hardware, configured explainers, metrics status, and run directory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AppConfig` | — | Active run config. |
| `model` | `Model` | — | Loaded model wrapper used to read `backend.hardware_label`. |

## `main`

```python
def main() -> None
```

Delegates to `raitap.run.__main__.main()` and exists so the CLI and module entry point share one callable.

## `extract_primary_tensor`

```python
def extract_primary_tensor(model_output: object) -> torch.Tensor
```

Selects the primary tensor from a raw model forward output. The helper prefers a raw tensor, then batch-like tensors in tuples or dicts, and finally the tensor with the largest `numel()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_output` | `object` | — | Model forward output that may be a tensor, sequence, or dict. |

Return type: `torch.Tensor`

Example:

```python
primary = extract_primary_tensor({"loss": loss, "logits": logits})
```

## Re-exported metrics helpers

```python
def metrics_prediction_pair(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
def resolve_metric_targets(predictions: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor
```

These are lazily re-exported from `raitap.metrics` through `__getattr__` in `src/raitap/run/__init__.py`.

## Output types

### `PredictionSummary`

```python
@dataclass(frozen=True)
class PredictionSummary:
    sample_index: int
    predicted_class: int
    confidence: float
    sample_id: str | None = None
    target_class: int | None = None
    correct: bool | None = None
```

### `RunOutputs`

```python
@dataclass(frozen=True)
class RunOutputs:
    explanations: list[ExplanationResult]
    visualisations: list[VisualisationResult]
    metrics: MetricsEvaluation | None
    forward_output: torch.Tensor
    sample_ids: list[str] | None = None
    targets: torch.Tensor | None = None
    prediction_summaries: tuple[PredictionSummary, ...] = ()
```

Use `RunOutputs` when you want to combine multiple stages, for example:

```python
outputs = run(cfg)
for explanation in outputs.explanations:
    print(explanation.run_dir)
```
