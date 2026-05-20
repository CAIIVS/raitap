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

**Use YAML + CLI when** your job is reproducible from disk, when you sweep parameters with Hydra multirun, or when you launch on Slurm. The CLI gives you `--multirun`, `--help`, dotted overrides, and persistent output directories. Configs in version control are the source of truth.

**Use Python when** you work in a notebook, embed RAITAP into another tool, want type checking against the dataclass schema, or build configs dynamically (e.g. a sweep generated in a `for` loop where each iteration depends on the previous result). The Python path skips Hydra's `chdir` and its logging hijack, so it composes cleanly with your own logging and working-directory conventions.

## API surface

The Python API is laid out so each module is the single owner of *both* its type contract (the schema dataclass) and its instances (the hydra-zen builders). Adding a new adapter is therefore strictly a single-file change inside its module — nothing else needs editing.

**Top-level — `from raitap import …`.** Orchestration-level only.

| Name         | Kind      | Why top-level                                |
| ------------ | --------- | -------------------------------------------- |
| `run`        | function  | runs an `AppConfig` through the orchestrator |
| `AppConfig`  | dataclass | root schema; consumed by `run`               |
| `Hardware`   | `StrEnum` | cross-cuts orchestrator + deps inference     |
| `raitap_log` | logger    | unified info/warn singleton                  |

**Per-module — `from raitap.<module> import …`.** Each module exposes its schema dataclass, its adapter builders, and any module-local enum.

| Module                | Schema                       | Builders                                                                                                                                                                                                     | Module enums                  |
| --------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| `raitap.models`       | `ModelConfig`                | —                                                                                                                                                                                                            | —                             |
| `raitap.data`         | `DataConfig`, `LabelsConfig` | —                                                                                                                                                                                                            | `LabelEncoding`, `IdStrategy` |
| `raitap.metrics`      | `MetricsConfig`              | `binary_classification`, `multiclass_classification`, `multilabel_classification`, `detection`                                                                                                               | —                             |
| `raitap.transparency` | `TransparencyConfig`         | `captum`, `shap`, `captum_image`, `captum_text`, `captum_time_series`, `shap_bar`, `shap_beeswarm`, `shap_force`, `shap_image`, `shap_waterfall`, `tabular_bar_chart`                                        | —                             |
| `raitap.robustness`   | `RobustnessConfig`           | `torchattacks`, `foolbox`, `marabou`, `image_pair`, `perturbation_heatmap`, `output_bounds_cohort`, `output_bounds_pinned`, `output_bounds_width_heatmap`, `output_bounds_margin_heatmap`, `verdict_summary` | —                             |
| `raitap.reporting`    | `ReportingConfig`            | `html`, `pdf`                                                                                                                                                                                                | —                             |
| `raitap.tracking`     | `TrackingConfig`             | `mlflow`                                                                                                                                                                                                     | —                             |

Each builder is a hydra-zen `builds()` dataclass; calling it returns an instance the orchestrator `instantiate`s. See {ref}`type-safety-map` for what is statically typed vs forwarded.

All names load lazily via :pep:`562` ``__getattr__`` so ``import raitap`` does not pull torch / Captum / torchattacks. `TYPE_CHECKING` blocks list every name statically so editor autocomplete still surfaces them.

## Install + quickstart

The Python equivalent of `raitap --demo` is roughly twenty lines. Build an `AppConfig`, pass it to `run`, read the structured `RunOutputs` (`raitap.pipeline.outputs.RunOutputs`) back:

```python
from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, LabelsConfig
from raitap.metrics import multiclass_classification
from raitap.models import ModelConfig
from raitap.robustness import image_pair, torchattacks
from raitap.transparency import captum, captum_image

cfg = AppConfig(
    hardware=Hardware.cpu,
    experiment_name="demo",
    model=ModelConfig(source="vit_b_32"),
    data=DataConfig(
        name="imagenet_samples",
        source="imagenet_samples",
        forward_batch_size=4,
        labels=LabelsConfig(
            source="imagenet_samples",
            id_column="image",
            column="label",
        ),
    ),
    metrics=multiclass_classification(num_classes=1000),
    transparency={
        "captum_ig": captum(
            algorithm="IntegratedGradients",
            call={"target": 0},
            visualisers=[captum_image()],
        ),
    },
    robustness={
        "pgd": torchattacks(
            algorithm="PGD",
            constructor={"eps": 0.03, "alpha": 0.005, "steps": 10},
            visualisers=[image_pair()],
        ),
    },
)

outputs = run(cfg, verbose=False)

first_explanation = outputs.explanations[0]
metric_values = outputs.metrics.result.metrics if outputs.metrics else {}
```

`verbose=False` suppresses the rich console summary panel but does not silence Python `logging`; configure the root logger yourself if you want quiet output.

Conversely, with `verbose=True` (the default) only the summary panel renders — per-step progress messages flow through Python `logging` and stay silent until you attach a handler. The CLI configures one via Hydra; the Python path leaves logging to you. A one-liner near the top of the script is enough:

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
```

### Auto-installing extras from Python

The `raitap` CLI walks the composed Hydra config before any heavy import, then installs the matching extras via `uv add` / `uv sync` (see <a href="../flags.html#flag-allow-project-edit"><code>--allow-project-edit</code></a>). The Python entry point gets the same flow with <a href="../flags.html#flag-allow-project-edit"><code>auto_install_deps=True</code></a>:

```python
from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, LabelsConfig
from raitap.metrics import multiclass_classification
from raitap.models import ModelConfig
from raitap.reporting import html
from raitap.robustness import image_pair, torchattacks
from raitap.transparency import captum, captum_image

cfg = AppConfig(
    hardware=Hardware.gpu,
    model=ModelConfig(source="vit_b_32"),
    metrics=multiclass_classification(num_classes=1000),
    transparency={"default": captum(algorithm="IntegratedGradients", call={"target": 0}, visualisers=[captum_image()])},
    robustness={"pgd": torchattacks(algorithm="PGD", constructor={"eps": 0.03}, visualisers=[image_pair()])},
    reporting=html(filename="report"),
)
run(cfg, auto_install_deps=True)
```

Why this works in a venv with **no extras installed yet**: every adapter module wraps its third-party library imports in `raitap.utils.lazy.lazy_import`, so importing `from raitap.metrics import multiclass_classification` does not pull `torchmetrics` at module load — only when you later instantiate the wrapped class. With `auto_install_deps=True`, `run` walks the cfg, infers the extras (it reads `_target_` strings the builders bake in), runs `uv add raitap[<extras>]`, and re-execs the script so the freshly-installed packages are visible when the pipeline actually invokes them. Idempotent: a sentinel env var short-circuits the second pass after the re-exec, so the same script line runs once and then becomes a no-op on the relaunch.

`auto_install_deps` is opt-in. Without it `run(cfg)` assumes the extras the config references are already installed — the typical case after a CLI bootstrap or a manual `uv sync`. A missing adapter library surfaces as the usual `ModuleNotFoundError` from the adapter import chain.

Pass <a href="../flags.html#flag-exec-global"><code>exec_global=True</code></a> together with `auto_install_deps=True` to consent to the bare-`pip install` fallback when no venv is active.

Each module exposes [hydra-zen `builds`](https://mit-ll-responsible-ai.github.io/hydra-zen/) factories — one per adapter — derived automatically from the class declaration. Import them from the relevant module:

```python
from raitap.transparency import captum, shap
from raitap.robustness import foolbox, torchattacks
from raitap.metrics import multiclass_classification, detection
```

These return *dataclass types* whose fields inherit the schema dataclass (`TransparencyConfig` / `RobustnessConfig` / `MetricsConfig`). Calling them with keyword arguments produces a config instance the orchestrator `instantiate`s — the same path Hydra takes when reading YAML. They are interchangeable with hand-written schema instances; choose whichever gives you better autocomplete in your editor.

## Kitchen-sink translation

The blocks below mirror, section by section, the YAML in {doc}`../examples/kitchen-sink`. Each `config-tabs` group renders the YAML on the left and the Python equivalent on the right.

### Model

```{config-tabs}
:yaml:
model:
  source: "./models/my-model.onnx"

:python:
from raitap.models import ModelConfig

model = ModelConfig(source="./models/my-model.onnx")
```

### Data

`DataConfig` nests a `LabelsConfig`. Both are plain dataclasses, so editor tooling will type-check the field names.

```{config-tabs}
:yaml:
data:
  name: "my-dataset"
  description: "Internal validation set"
  source: "./data/images"
  forward_batch_size: 32
  labels:
    source: "./data/labels.csv"
    id_column: "image"
    column: "label"
    encoding: "index"

:python:
from raitap.data import DataConfig, LabelEncoding, LabelsConfig

data = DataConfig(
    name="my-dataset",
    description="Internal validation set",
    source="./data/images",
    forward_batch_size=32,
    labels=LabelsConfig(
        source="./data/labels.csv",
        id_column="image",
        column="label",
        encoding=LabelEncoding.index,  # or the literal string "index"
    ),
)
```

### Metrics

```{config-tabs}
:yaml:
metrics:
  _target_: "MulticlassClassificationMetrics"
  num_classes: 7
  average: "macro"

:python:
from raitap.metrics import multiclass_classification

metrics = multiclass_classification(num_classes=7, average="macro")
```

The hydra-zen builder is the recommended path here because `MulticlassClassificationMetrics` exposes many optional kwargs (`average`, `ignore_index`, …) that the bare `MetricsConfig` discriminator does not type explicitly. `populate_full_signature=True` lifts the full constructor signature onto the dataclass, so your editor will autocomplete every kwarg. Pick the matching builder (`binary_classification`, `multiclass_classification`, `multilabel_classification`, `detection`) for the task you're configuring.

### Transparency

`AppConfig.transparency` is a `dict[str, TransparencyConfig]`. The dict key (`captum_ig`, `shap_gradient`) is the *run name*; it shows up in reports and tracking artefacts so different explainer configurations stay distinguishable.

```{config-tabs}
:yaml:
transparency:
  captum_ig:
    _target_: "CaptumExplainer"
    algorithm: "IntegratedGradients"
    constructor: {}
    call:
      target: 0
      baselines:
        source: "./data/baselines"
        n_samples: 8
    visualisers:
      - _target_: "CaptumImageVisualiser"
        constructor:
          method: "blended_heat_map"
          sign: "all"
          show_colorbar: true
          title: "Integrated gradients"
          include_original_image: true
        call:
          max_samples: 4
          show_sample_names: true
  shap_gradient:
    _target_: "ShapExplainer"
    algorithm: "GradientExplainer"
    constructor:
      local_smoothing: 0.0
    call:
      target: 0
      nsamples: 10
      background_data:
        source: "./data/background"
        n_samples: 32
    raitap:
      batch_size: 1
      progress_desc: "SHAP batches"
    visualisers:
      - _target_: "ShapImageVisualiser"
        constructor:
          max_samples: 2

:python:
from raitap.transparency import captum, captum_image, shap, shap_image

transparency = {
    "captum_ig": captum(
        algorithm="IntegratedGradients",
        call={
            "target": 0,
            "baselines": {"source": "./data/baselines", "n_samples": 8},
        },
        visualisers=[
            captum_image(
                method="blended_heat_map",
                sign="all",
                show_colorbar=True,
                title="Integrated gradients",
                include_original_image=True,
                call={"max_samples": 4, "show_sample_names": True},
            ),
        ],
    ),
    "shap_gradient": shap(
        algorithm="GradientExplainer",
        constructor={"local_smoothing": 0.0},
        call={
            "target": 0,
            "nsamples": 10,
            "background_data": {"source": "./data/background", "n_samples": 32},
        },
        raitap={"batch_size": 1, "progress_desc": "SHAP batches"},
        visualisers=[shap_image(max_samples=2)],
    ),
}
```

Visualisers expose a builder per class (`captum_image`, `shap_image`, `image_pair`, `perturbation_heatmap`, …). Flat constructor kwargs map to the wrapped `__init__`; pass `call={...}` for render-time options. No `_target_` strings needed.

### Robustness

Robustness uses the same shape: `dict[str, RobustnessConfig]`, named runs, list of visualisers.

```{config-tabs}
:yaml:
robustness:
  pgd:
    _target_: "TorchattacksAssessor"
    algorithm: "PGD"
    constructor:
      eps: 0.03
      alpha: 0.0078
      steps: 10
    visualisers:
      - _target_: "ImagePairVisualiser"
        constructor:
          max_samples: 4
  linf_pgd:
    _target_: "FoolboxAssessor"
    algorithm: "LinfPGD"
    constructor:
      rel_stepsize: 0.025
      steps: 40
    call:
      eps: 0.03
    visualisers:
      - _target_: "PerturbationHeatmapVisualiser"

:python:
from raitap.robustness import foolbox, image_pair, perturbation_heatmap, torchattacks

robustness = {
    "pgd": torchattacks(
        algorithm="PGD",
        constructor={"eps": 0.03, "alpha": 0.0078, "steps": 10},
        visualisers=[image_pair(max_samples=4)],
    ),
    "linf_pgd": foolbox(
        algorithm="LinfPGD",
        constructor={"rel_stepsize": 0.025, "steps": 40},
        call={"eps": 0.03},
        visualisers=[perturbation_heatmap()],
    ),
}
```

### Reporting

```{config-tabs}
:yaml:
reporting:
  _target_: "HTMLReporter"
  filename: "report"
  multirun_report: true
  show_original_per_explainer: false
  show_redundant_robustness_panels: false

:python:
from raitap.reporting import html

reporting = html(
    filename="report",
    multirun_report=True,
    show_original_per_explainer=False,
    show_redundant_robustness_panels=False,
)
```

Leaving `reporting=None` on the `AppConfig` (or omitting the section in YAML) disables report generation entirely — useful from notebooks where you inspect the in-memory `RunOutputs` directly.

### Tracking

```{config-tabs}
:yaml:
tracking:
  _target_: "MLFlowTracker"
  output_forwarding_url: "http://127.0.0.1:5001"
  log_model: false
  open_when_done: true

:python:
from raitap.tracking import mlflow

tracking = mlflow(
    output_forwarding_url="http://127.0.0.1:5001",
    log_model=False,
    open_when_done=True,
)
```

### Putting it together

```python
from raitap import AppConfig, Hardware, run

cfg = AppConfig(
    hardware=Hardware.gpu,
    experiment_name="My Experiment",
    model=model,
    data=data,
    transparency=transparency,
    robustness=robustness,
    metrics=metrics,
    tracking=tracking,
    reporting=reporting,
)
outputs = run(cfg)
```

## Translation rules

| YAML pattern                                                     | Python builder                                                                                                                                                                                                       |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_target_: CaptumExplainer` + `algorithm: IntegratedGradients`   | `captum(algorithm="IntegratedGradients", ...)` (the `_target_` is baked in)                                                                                                                                          |
| `defaults: [raitap_schema, _self_]`                              | Not needed — `AppConfig` already *is* the schema. The defaults entry is a Hydra-only construct.                                                                                                                      |
| Group/name selection (`transparency: captum` + dict key in YAML) | Use the dict key on the Python side too: `transparency={"my_run": captum(algorithm=...)}`.                                                                                                                           |
| List of visualisers                                              | One builder per visualiser (`captum_image`, `image_pair`, …): flat constructor kwargs, optional `call={...}` for render-time options. `visualisers=[captum_image(max_samples=4, call={"show_sample_names": True})]`. |
| `MISSING` defaults                                               | Builder kwargs are required-or-optional based on the wrapped constructor signature; your editor surfaces the missing ones.                                                                                           |
| CLI overrides (`+foo.bar=baz`)                                   | Mutate the dataclass: `cfg.transparency["my_run"].call["target"] = 1`. Builders return dataclasses, so attribute assignment works.                                                                                   |

(type-safety-map)=

## Type safety map

The schema is a deliberate mix of strict and forwarded. Knowing which fields are which avoids head-scratching about why one typo is caught immediately and another only at run time.

**Fully typed**

- `hardware: Hardware`, `data.labels.encoding: LabelEncoding`, `data.labels.id_strategy: IdStrategy`, `data.labels.kind: LabelKind` — all four are `enum.StrEnum` subclasses. Imports follow the module-ownership rule: `from raitap import Hardware`, `from raitap.data import LabelEncoding, IdStrategy`, `from raitap.data.types import LabelKind`. **Pass the enum member** (`Hardware.cpu`) — typos like `Hardware.cpuu` fail at import time and Pyright/your editor catch them immediately. The raw string form (`"cpu"`) still parses at runtime via OmegaConf coercion, but it bypasses static type-checking — defeating the main reason to use the Python path over YAML.
- The nested dataclass dicts on `AppConfig.transparency` and `AppConfig.robustness` — keys are arbitrary user-chosen strings, values must be `TransparencyConfig` / `RobustnessConfig` instances (or dicts with the right keys).
- All scalar fields on `ModelConfig`, `DataConfig`, `LabelsConfig`, `MetricsConfig`, `TrackingConfig`, `ReportingConfig` are checked by OmegaConf's structured-config validation when the orchestrator boots.

**Library-forwarded kwargs (unchecked at schema time)**

- `TransparencyConfig.constructor` / `.call` / `.raitap` — dicts forwarded verbatim to the underlying explainer library (Captum's `IntegratedGradients(...)`, SHAP's `GradientExplainer(...)`, the corresponding `.attribute()` / `.shap_values()` call).
- `RobustnessConfig.constructor` / `.call` / `.raitap` — same story for torchattacks / Foolbox / Marabou.

The hydra-zen builders in `raitap.api` (`captum`, `shap`, `torchattacks`, `foolbox`, plus the per-task metrics builders `binary_classification`, `multiclass_classification`, `multilabel_classification`, `detection`) inherit the corresponding schema dataclass via `builds_bases=`, so they accept every field on `TransparencyConfig` / `RobustnessConfig` / `MetricsConfig` (`algorithm`, `constructor`, `call`, `raitap`, `visualisers`). The classification and detection builders additionally use `populate_full_signature=True` to surface the full adapter `__init__` signature (`average`, `num_labels`, `ignore_index`, `iou`, …). For Captum / SHAP / torchattacks / Foolbox, the underlying constructor takes `**kwargs`, so per-library kwargs (`eps=`, `steps=`, `target=`) stay inside the `constructor` / `call` dict — your editor autocompletes the schema fields, not the library kwargs.

(runoutputs-shape)=

## RunOutputs shape

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

When `reporting` is configured these fields flow through `HTMLReporter` and end up on disk under the reporting output directory; when `tracking` is configured they are logged as MLflow artefacts. From Python you can read them straight out of the returned object without enabling either subsystem.

## CLI ↔ Python parity

Single-run YAML configs round-trip cleanly: the Python `AppConfig` you build by hand and the dataclass that Hydra materialises from `demo.yaml` drive the orchestrator identically (the test suite asserts this — see `test_run_parity_with_yaml_demo` in `src/raitap/tests/test_api.py`).

Multirun sweeps stay CLI-only. Hydra's `--multirun` machinery, sweep plugins, and the `${hydra.sweep.dir}` resolver have no in-Python equivalent. When you need a sweep from Python, build the configs yourself and call `run` per config:

```python
from copy import deepcopy

from raitap import run

base = _build_base_config()  # an AppConfig from earlier in this page

results = []
for eps in (0.01, 0.03, 0.06, 0.1):
    cfg = deepcopy(base)
    cfg.robustness["pgd"].constructor["eps"] = eps
    cfg.experiment_name = f"pgd-eps={eps}"
    results.append((eps, run(cfg, verbose=False)))

for eps, outputs in results:
    print(eps, outputs.metrics.result.metrics if outputs.metrics else None)
```

For anything more elaborate than a one-axis sweep — e.g. cross-product over `(eps, steps)`, conditional skips, dependency between consecutive runs — a Python `for` loop with `deepcopy` is both simpler and faster to read than the equivalent `--multirun` invocation. The trade-off is that you lose Hydra's automatic per-run output directories; manage those yourself if you need them.
