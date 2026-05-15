---
orphan: true
---

# Python API

RAITAP can be driven from Python directly, without YAML files or the CLI. The Python entry point shares the orchestrator, schema, and side-effects with `raitap --config-name ...`; only the *front door* differs.

This page is the canonical translation reference between YAML configs and the Python API. Every snippet pair below uses the `config-tabs` directive so you can flip the whole page between the two surfaces at once.

## When to use which

**Use YAML + CLI when** your job is reproducible from disk, when you sweep parameters with Hydra multirun, or when you launch on Slurm. The CLI gives you `--multirun`, `--help`, dotted overrides, and persistent output directories. Configs in version control are the source of truth.

**Use Python when** you work in a notebook, embed RAITAP into another tool, want type checking against the dataclass schema, or build configs dynamically (e.g. a sweep generated in a `for` loop where each iteration depends on the previous result). The Python path skips Hydra's `chdir` and its logging hijack, so it composes cleanly with your own logging and working-directory conventions.

## Install + quickstart

The Python equivalent of `raitap --demo` is roughly twenty lines. Build an `AppConfig` (`raitap.configs.schema.AppConfig`), pass it to `raitap.run`, read the structured `RunOutputs` (`raitap.pipeline.outputs.RunOutputs`) back:

```python
from raitap import AppConfig, run
from raitap.configs.schema import (
    DataConfig,
    LabelsConfig,
    MetricsConfig,
    ModelConfig,
    RobustnessConfig,
    TransparencyConfig,
)
from raitap.robustness import image_pair
from raitap.transparency import captum_image

cfg = AppConfig(
    hardware="cpu",
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
    metrics=MetricsConfig(_target_="ClassificationMetrics", task="multiclass"),
    transparency={
        "default": TransparencyConfig(
            _target_="CaptumExplainer",
            algorithm="IntegratedGradients",
            call={"target": 0},
            visualisers=[captum_image()],
        )
    },
    robustness={
        "pgd": RobustnessConfig(
            _target_="TorchattacksAssessor",
            algorithm="PGD",
            constructor={"eps": 0.03, "alpha": 0.005, "steps": 10},
            visualisers=[image_pair()],
        )
    },
)

outputs = run(cfg, verbose=False)

first_explanation = outputs.explanations[0]
metric_values = outputs.metrics.result.metrics if outputs.metrics else {}
```

`verbose=False` suppresses the rich console summary panel but does not silence Python `logging`; configure the root logger yourself if you want quiet output.

Each module exposes [hydra-zen `builds`](https://mit-ll-responsible-ai.github.io/hydra-zen/) factories — one per adapter — derived automatically from the class declaration. Import them from the relevant module:

```python
from raitap.transparency import captum, shap
from raitap.robustness import foolbox, torchattacks
from raitap.metrics import classification
```

These return *dataclass types* whose fields inherit the schema dataclass (`TransparencyConfig` / `RobustnessConfig` / `MetricsConfig`). Calling them with keyword arguments produces a config instance the orchestrator `instantiate`s — the same path Hydra takes when reading YAML. They are interchangeable with hand-written schema instances; choose whichever gives you better autocomplete in your editor.

## Kitchen-sink translation

The blocks below mirror, section by section, the YAML in {doc}`kitchen-sink`. Each `config-tabs` group renders the YAML on the left and the Python equivalent on the right.

### Model

```{config-tabs}
:yaml:
model:
  source: "./models/my-model.onnx"

:python:
from raitap.configs.schema import ModelConfig

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
from raitap.configs.schema import DataConfig, LabelsConfig
from raitap.types import LabelEncoding

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
  _target_: "ClassificationMetrics"
  task: "multiclass"
  num_classes: 7
  average: "macro"

:python:
from raitap.metrics import classification

metrics = classification(task="multiclass", num_classes=7, average="macro")
```

The hydra-zen builder is the recommended path here because `ClassificationMetrics` exposes many optional kwargs (`average`, `ignore_index`, …) that `MetricsConfig` does not type explicitly. `populate_full_signature=True` lifts the full constructor signature onto the dataclass, so your editor will autocomplete every kwarg.

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
from raitap.configs.schema import TransparencyConfig

transparency = {
    "captum_ig": TransparencyConfig(
        _target_="CaptumExplainer",
        algorithm="IntegratedGradients",
        call={
            "target": 0,
            "baselines": {"source": "./data/baselines", "n_samples": 8},
        },
        visualisers=[
            {
                "_target_": "CaptumImageVisualiser",
                "constructor": {
                    "method": "blended_heat_map",
                    "sign": "all",
                    "show_colorbar": True,
                    "title": "Integrated gradients",
                    "include_original_image": True,
                },
                "call": {"max_samples": 4, "show_sample_names": True},
            }
        ],
    ),
    "shap_gradient": TransparencyConfig(
        _target_="ShapExplainer",
        algorithm="GradientExplainer",
        constructor={"local_smoothing": 0.0},
        call={
            "target": 0,
            "nsamples": 10,
            "background_data": {"source": "./data/background", "n_samples": 32},
        },
        raitap={"batch_size": 1, "progress_desc": "SHAP batches"},
        visualisers=[
            {"_target_": "ShapImageVisualiser", "constructor": {"max_samples": 2}}
        ],
    ),
}
```

Visualisers stay as plain dicts even when the explainer is built via a typed dataclass — they are heterogeneous (different `_target_` per entry) and the orchestrator resolves them through Hydra `instantiate` at run time.

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
from raitap.configs.schema import RobustnessConfig
from raitap.robustness import perturbation_heatmap

robustness = {
    "pgd": RobustnessConfig(
        _target_="TorchattacksAssessor",
        algorithm="PGD",
        constructor={"eps": 0.03, "alpha": 0.0078, "steps": 10},
        visualisers=[
            {"_target_": "ImagePairVisualiser", "constructor": {"max_samples": 4}}
        ],
    ),
    "linf_pgd": RobustnessConfig(
        _target_="FoolboxAssessor",
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
from raitap.configs.schema import ReportingConfig

reporting = ReportingConfig(
    _target_="HTMLReporter",
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
from raitap.configs.schema import TrackingConfig

tracking = TrackingConfig(
    _target_="MLFlowTracker",
    output_forwarding_url="http://127.0.0.1:5001",
    log_model=False,
    open_when_done=True,
)
```

### Putting it together

```python
from raitap import AppConfig, run

cfg = AppConfig(
    hardware="gpu",
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

The four patterns below cover every shape you'll meet when porting a YAML config to Python.

| YAML pattern | Python (dict shape) | Python (hydra-zen builder) |
| --- | --- | --- |
| `_target_: CaptumExplainer` | `{"_target_": "CaptumExplainer", ...}` or `TransparencyConfig(_target_="CaptumExplainer", ...)` | `captum(algorithm="IntegratedGradients", ...)` (the `_target_` is baked in) |
| `defaults: [raitap_schema, _self_]` | Not needed — `AppConfig` already *is* the schema. The defaults entry is a Hydra-only construct. | Same — builders return dataclass types bound to the right `_target_`. |
| Group/name selection (`transparency: captum`) | Set the key on the dict yourself: `transparency={"default": TransparencyConfig(...)}`. The key is the run name. | `transparency={"default": captum(algorithm=...)}` works identically. |
| List of visualisers | List of dicts: `visualisers=[{"_target_": "...", "constructor": {...}}]`. Required for visualisers with separate `constructor:` / `call:` blocks. | Builder per visualiser, e.g. `from raitap.transparency import captum_image` / `from raitap.robustness import image_pair`; call as `visualisers=[captum_image(max_samples=4)]`. Builders only expose the `__init__` kwargs (constructor) — fall back to the dict shape when you need a `call:` block. |
| `MISSING` defaults | Fields default to `omegaconf.MISSING` so omitting `_target_` / `algorithm` raises at validation time. Provide both explicitly. | Builder kwargs are required-or-optional based on the wrapped constructor signature; let your editor surface the missing ones. |
| CLI overrides (`+foo.bar=baz`) | Mutate the dataclass: `cfg.transparency["default"].call["target"] = 1`. | Same — builders produce dataclasses, so attribute assignment works. |

## Type safety map

The schema is a deliberate mix of strict and forwarded. Knowing which fields are which avoids head-scratching about why one typo is caught immediately and another only at run time.

**Fully typed**

- `hardware: Hardware`, `data.labels.encoding: LabelEncoding`, `data.labels.id_strategy: IdStrategy`, `metrics.task: Task` — all four are `enum.StrEnum` subclasses defined in `raitap.types`. Pass either the enum member (`Hardware.cpu`) or its string value (`"cpu"`); OmegaConf validates the latter against the member name.
- The nested dataclass dicts on `AppConfig.transparency` and `AppConfig.robustness` — keys are arbitrary user-chosen strings, values must be `TransparencyConfig` / `RobustnessConfig` instances (or dicts with the right keys).
- All scalar fields on `ModelConfig`, `DataConfig`, `LabelsConfig`, `MetricsConfig`, `TrackingConfig`, `ReportingConfig` are checked by OmegaConf's structured-config validation when the orchestrator boots.

**Library-forwarded kwargs (unchecked at schema time)**

- `TransparencyConfig.constructor` / `.call` / `.raitap` — dicts forwarded verbatim to the underlying explainer library (Captum's `IntegratedGradients(...)`, SHAP's `GradientExplainer(...)`, the corresponding `.attribute()` / `.shap_values()` call).
- `RobustnessConfig.constructor` / `.call` / `.raitap` — same story for torchattacks / Foolbox / Marabou.

The hydra-zen builders in `raitap.api` (`captum`, `shap`, `torchattacks`, `foolbox`, `classification_metrics`) inherit the corresponding schema dataclass via `builds_bases=`, so they accept every field on `TransparencyConfig` / `RobustnessConfig` / `MetricsConfig` (`algorithm`, `constructor`, `call`, `raitap`, `visualisers`). `classification_metrics` additionally uses `populate_full_signature=True` to surface the full `ClassificationMetrics.__init__` signature (`average`, `num_labels`, `ignore_index`). For Captum / SHAP / torchattacks / Foolbox, the underlying constructor takes `**kwargs`, so per-library kwargs (`eps=`, `steps=`, `target=`) stay inside the `constructor` / `call` dict — your editor autocompletes the schema fields, not the library kwargs.

## RunOutputs shape

`raitap.run` returns a frozen `RunOutputs` dataclass:

| Field | Type | Meaning |
| --- | --- | --- |
| `explanations` | `list[ExplanationResult]` | One result per `(transparency_run, sample_batch)` pair; carries the attribution tensor and metadata. |
| `visualisations` | `list[VisualisationResult]` | Rendered transparency outputs (images, HTML fragments) ready for reporting. |
| `metrics` | `MetricsEvaluation \| None` | Aggregated classification metrics. `None` when `metrics` is unconfigured. |
| `forward_output` | `torch.Tensor` | Raw output of `model(data)` — the source for both predictions and metric inputs. |
| `sample_ids` | `list[str] \| None` | Stable ids aligned with `forward_output` rows; `None` when the data source doesn't supply them. |
| `targets` | `torch.Tensor \| None` | Ground-truth labels when labels are configured. |
| `prediction_summaries` | `tuple[PredictionSummary, ...]` | Per-sample `(index, predicted_class, confidence, sample_id, target_class, correct)`. |
| `robustness_results` | `list[RobustnessResult]` | One per `(robustness_run, sample_batch)`; carries adversarial tensors and per-sample success flags. |
| `robustness_visualisations` | `list[RobustnessVisualisationResult]` | Rendered robustness outputs (image pairs, heat maps). |

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
