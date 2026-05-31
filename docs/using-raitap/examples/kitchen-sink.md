---
title: "Kitchen-sink example"
description: "The example below shows a complete configuration with all top-level modules populated."
myst:
  html_meta:
    "description": "The example below shows a complete configuration with all top-level modules populated."
---

# Kitchen-sink example

The example below shows a complete configuration with all top-level modules populated.

If you want to learn how to write such a config, see the {doc}`../configuration/general` guide. The {doc}`../configuration/python-api` page covers the equivalent programmatic surface.

```{config-tabs}
:yaml:
hardware: "gpu"
experiment_name: "My Experiment"

model:
  source: "./models/my-model.onnx"

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

transparency:
  captum_ig:
    _target_: "CaptumExplainer"
    algorithm: "IntegratedGradients"
    constructor: {}
    call:
      target: 0
    raitap:
      baseline:
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
    raitap:
      baseline:
        source: "./data/background"
        n_samples: 32
      batch_size: 1
      progress_desc: "SHAP batches"
    visualisers:
      - _target_: "ShapImageVisualiser"
        constructor:
          max_samples: 2

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

metrics:
  _target_: "MulticlassClassificationMetrics"
  num_classes: 7
  average: "macro"
  ignore_index: null

tracking:
  _target_: "MLFlowTracker"
  output_forwarding_url: "http://127.0.0.1:5001"
  log_model: false
  open_when_done: true

reporting:
  _target_: "HTMLReporter"
  filename: "report"
  multirun_report: true
  show_original_per_explainer: false
  show_redundant_robustness_panels: false

:python:
from raitap import AppConfig, Hardware
from raitap.data import DataConfig, LabelEncoding, LabelsConfig
from raitap.metrics import multiclass_classification
from raitap.models import ModelConfig
from raitap.reporting import html as html_report
from raitap.robustness import foolbox, image_pair, perturbation_heatmap, torchattacks
from raitap.tracking import mlflow
from raitap.transparency import captum, captum_image, shap, shap_image

config = AppConfig(
    hardware=Hardware.gpu,
    experiment_name="My Experiment",
    model=ModelConfig(source="./models/my-model.onnx"),
    data=DataConfig(
        name="my-dataset",
        description="Internal validation set",
        source="./data/images",
        forward_batch_size=32,
        labels=LabelsConfig(
            source="./data/labels.csv",
            id_column="image",
            column="label",
            encoding=LabelEncoding.index,
        ),
    ),
    transparency={
        "captum_ig": captum(
            algorithm="IntegratedGradients",
            call={"target": 0},
            raitap={"baseline": {"source": "./data/baselines", "n_samples": 8}},
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
            call={"target": 0, "nsamples": 10},
            raitap={
                "baseline": {"source": "./data/background", "n_samples": 32},
                "batch_size": 1,
                "progress_desc": "SHAP batches",
            },
            visualisers=[shap_image(max_samples=2)],
        ),
    },
    robustness={
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
    },
    metrics=multiclass_classification(
        num_classes=7,
    ),
    tracking=mlflow(
        output_forwarding_url="http://127.0.0.1:5001",
        log_model=False,
        open_when_done=True,
    ),
    reporting=html_report(
        filename="report",
        multirun_report=True,
        show_original_per_explainer=False,
        show_redundant_robustness_panels=False,
    ),
)
```

## Notes

The kitchen-sink config references user-supplied artefacts
(`./models/my-model.onnx`, `./data/images`, `./data/labels.csv`,
`./data/baselines`, `./data/background`) and a local MLflow server at
`127.0.0.1:5001`.

## Expected output

```text
outputs/<date>/<time>/
├── metrics/{metrics.json, artifacts.json, metadata.json, metrics_overview.png}
├── transparency/
│   ├── captum_ig/{attributions.pt, CaptumImageVisualiser_0.png, metadata.json}
│   └── shap_gradient/{attributions.pt, ShapImageVisualiser_0.png, metadata.json}
├── robustness/
│   ├── pgd/{robustness_data.pt, ImagePairVisualiser_0.png, metadata.json}
│   └── linf_pgd/{robustness_data.pt, PerturbationHeatmapVisualiser_0.png, metadata.json}
├── tracking/{run_id.txt, mlflow.log}
└── reports/{report.html, report.zip, _assets/…}
```

MLflow artefacts land on the tracking server itself
(`http://127.0.0.1:5001`). The `tracking/` directory only carries the
local hand-off metadata.
