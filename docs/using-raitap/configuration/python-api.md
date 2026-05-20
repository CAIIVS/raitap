---
orphan: true
title: "Python API"
description: "RAITAP can be driven from Python directly, without YAML files or the CLI. The Python entry point shares the orchestrator, schema, and side-effects with raitap --config-name ...; only the front door differs."
myst:
  html_meta:
    "description": "RAITAP can be driven from Python directly, without YAML files or the CLI. The Python entry point shares the orchestrator, schema, and side-effects with raitap --config-name ...; only the front door differs."
---

# Python API

This page explains how to use RAITAP from Python directly, without YAML files or the CLI.

## When to use which

**Use YAML + CLI when** your job is reproducible from disk, when you **sweep parameters with Hydra multirun**, or when you launch on **Slurm**. The CLI gives you `--multirun`, `--help`, dotted overrides, and persistent output directories. Configs in version control are the source of truth.

**Use Python when** you work in a **notebook**, embed RAITAP into another tool, want **type-checking** against the dataclass schema, or **build configs dynamically** (e.g. a sweep generated in a `for` loop where each iteration depends on the previous result). The Python path skips Hydra's `chdir` and its logging hijack, so it composes cleanly with your own logging and working-directory conventions.

## API surface

The general use objects are exported by the `raitap` package:

```python
from raitap import AppConfig, Hardware, run
```

The module-specific objects are exported by the respective modules:

```python
from raitap.models import ModelConfig
from raitap.data import DataConfig, LabelsConfig
from raitap.metrics import multiclass_classification
from raitap.robustness import image_pair, torchattacks
from raitap.transparency import captum, captum_image
```

## Install + quickstart

The Python equivalent of `raitap --demo` is roughly twenty lines. Build an `AppConfig`, pass it to `run`, read the structured `RunOutputs` (`raitap.pipeline.outputs.RunOutputs`) back:

```python
from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, LabelsConfig
from raitap.metrics import multiclass_classification
from raitap.models import ModelConfig
from raitap.robustness import image_pair, torchattacks
from raitap.transparency import captum, captum_image

cfg = # config code, omitted

outputs = run(cfg, verbose=False)
```

`verbose=False` suppresses the rich console summary panel but does not silence Python `logging`; configure the root logger yourself if you want quiet output.

Conversely, with `verbose=True` (the default) only the summary panel renders — per-step progress messages flow through Python `logging` and stay silent until you attach a handler. The CLI configures one via Hydra; the Python path leaves logging to you. A one-liner near the top of the script is enough:

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
```

### Auto-installing extras from Python

Pass the corresponding flag to the `run` function:

```python
# imports omitted

cfg = # config code, omitted

run(cfg, auto_install_deps=True)
```

`auto_install_deps` is opt-in. Without it `run(cfg)` assumes the extras the config references are already installed — the typical case after a CLI bootstrap or a manual `uv sync`. A missing adapter library surfaces as the usual `ModuleNotFoundError` from the adapter import chain.

## Translation rules

| YAML pattern                                                     | Python builder                                                                                                                                                                                                       |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_target_: CaptumExplainer` + `algorithm: IntegratedGradients`   | `captum(algorithm="IntegratedGradients", ...)` (the `_target_` is baked in)                                                                                                                                          |
| `defaults: [raitap_schema, _self_]`                              | Not needed — `AppConfig` already *is* the schema. The defaults entry is a Hydra-only construct.                                                                                                                      |
| Group/name selection (`transparency: captum` + dict key in YAML) | Use the dict key on the Python side too: `transparency={"my_run": captum(algorithm=...)}`.                                                                                                                           |
| List of visualisers                                              | One builder per visualiser (`captum_image`, `image_pair`, …): flat constructor kwargs, optional `call={...}` for render-time options. `visualisers=[captum_image(max_samples=4, call={"show_sample_names": True})]`. |
| `MISSING` defaults                                               | Builder kwargs are required-or-optional based on the wrapped constructor signature; your editor surfaces the missing ones.                                                                                           |
| CLI overrides (`+foo.bar=baz`)                                   | Mutate the dataclass: `cfg.transparency["my_run"].call["target"] = 1`. Builders return dataclasses, so attribute assignment works.                                                                                   |

See {doc}`../examples/index` for translation examples.

(type-safety-map)=

## Type safety

Due to how Hydra works, only some fields are typed.

### Fully typed

- `hardware: Hardware`, `data.labels.encoding: LabelEncoding`, `data.labels.id_strategy: IdStrategy`, `data.labels.kind: LabelKind` — all four are `enum.StrEnum` subclasses.
- The nested dataclass dicts on `AppConfig.transparency` and `AppConfig.robustness` — keys are arbitrary user-chosen strings, values must be `TransparencyConfig` / `RobustnessConfig` instances (or dicts with the right keys).
- All scalar fields on `ModelConfig`, `DataConfig`, `LabelsConfig`, `MetricsConfig`, `TrackingConfig`, `ReportingConfig` are checked by OmegaConf's structured-config validation when the orchestrator boots.

Library-forwarded kwargs (unchecked at schema time):

- `TransparencyConfig.constructor` / `.call` / `.raitap` — dicts forwarded verbatim to the underlying explainer library (Captum's `IntegratedGradients(...)`, SHAP's `GradientExplainer(...)`, the corresponding `.attribute()` / `.shap_values()` call).
- `RobustnessConfig.constructor` / `.call` / `.raitap` — same story for torchattacks / Foolbox / Marabou.

(runoutputs-shape)=

## `RunOutputs` shape

`raitap.run` returns a frozen `RunOutputs` dataclass:

| Field                       | Type                                  | Meaning                                                                                              |
| --------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `explanations`              | `list[ExplanationResult]`             | One result per `(transparency_run, sample_batch)` pair; carries the attribution tensor and metadata. |
| `visualisations`            | `list[VisualisationResult]`           | Rendered transparency outputs (images, HTML fragments) ready for reporting.                          |
| `metrics`                   | `MetricsEvaluation \| None`           | Aggregated classification metrics. `None` when `metrics` is unconfigured.                            |
| `forward_output`            | `torch.Tensor`                        | Raw output of `model(data)` — the source for both predictions and metric inputs.                     |
| `sample_ids`                | `list[str] \| None`                   | Stable ids aligned with `forward_output` rows; `None` when the data source doesn't supply them.      |
| `targets`                   | `torch.Tensor \| None`                | Ground-truth labels when labels are configured.                                                      |
| `prediction_summaries`      | `tuple[PredictionSummary, ...]`       | Per-sample `(index, predicted_class, confidence, sample_id, target_class, correct)`.                 |
| `robustness_results`        | `list[RobustnessResult]`              | One per `(robustness_run, sample_batch)`; carries adversarial tensors and per-sample success flags.  |
| `robustness_visualisations` | `list[RobustnessVisualisationResult]` | Rendered robustness outputs (image pairs, heat maps).                                                |

## Multiruns in Python

Only the CLI can benefit from Hydra's multirun feature (see {ref}`multirun`), but you can recreate the loop in Python directly:

```python
from copy import deepcopy

from raitap import run

cfg = # omitted, see example above

results = []
for eps in (0.01, 0.03, 0.06, 0.1):
    copied_cfg = deepcopy(cfg)
    copied_cfg.robustness["pgd"].constructor["eps"] = eps
    copied_cfg.experiment_name = f"pgd-eps={eps}"
    results.append((eps, run(copied_cfg, verbose=False)))

for eps, outputs in results:
    print(eps, outputs.metrics.result.metrics if outputs.metrics else None)
```
