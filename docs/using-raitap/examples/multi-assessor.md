---
title: "Multi-assessor · two attacks against one model"
description: "Two robustness assessors under one run — Torchattacks PGD (iterative L∞ attack) and Torchattacks FGSM (single-step). Compares attack success rates and per-sample distances side by side in the HTML report."
myst:
  html_meta:
    "description": "Two robustness assessors under one run — Torchattacks PGD (iterative L∞ attack) and Torchattacks FGSM (single-step). Compares attack success rates and per-sample distances side by side in the HTML report."
---

# Multi-assessor · two attacks against one model

```{recipe}
:summary: Two robustness assessors under one run — Torchattacks PGD (iterative L∞ attack) and Torchattacks FGSM (single-step). Compares attack success rates and per-sample distances side by side in the HTML report.

:yaml:
defaults:
  - raitap_schema
  - reporting: html
  - metrics: multiclass_classification
  - _self_

hardware: gpu
experiment_name: multi-assessor

model:
  source: vit_b_32

data:
  name: imagenet_samples
  source: imagenet_samples
  forward_batch_size: 4
  labels:
    source: imagenet_samples
    id_column: image
    column: label

metrics:
  num_classes: 1000
  # `micro` aggregates across all samples before averaging — equals overall
  # accuracy on multiclass single-label runs. Good when every sample carries
  # equal weight (this 4-image demo).
  average: micro

transparency:
  default:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    call:
      target: 0
    visualisers:
      - _target_: CaptumImageVisualiser

robustness:
  pgd:
    _target_: TorchattacksAssessor
    algorithm: PGD
    constructor:
      eps: 0.03
      alpha: 0.005
      steps: 10
    visualisers:
      - _target_: ImagePairVisualiser
  fgsm:
    _target_: TorchattacksAssessor
    algorithm: FGSM
    constructor:
      eps: 0.03
    visualisers:
      - _target_: ImagePairVisualiser

:python:
from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, LabelsConfig
from raitap.metrics import multiclass_classification
from raitap.models import ModelConfig
from raitap.reporting import html
from raitap.robustness import image_pair, torchattacks
from raitap.transparency import captum, captum_image

cfg = AppConfig(
    hardware=Hardware.gpu,
    experiment_name="multi-assessor",
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
    metrics=multiclass_classification(num_classes=1000, average="micro"),
    transparency={
        "default": captum(
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
        "fgsm": torchattacks(
            algorithm="FGSM",
            constructor={"eps": 0.03},
            visualisers=[image_pair()],
        ),
    },
    reporting=html(filename="report"),
)
outputs = run(cfg, auto_install_deps=True)

:output:
outputs/<date>/<time>/
├── metrics/{metrics.json, artifacts.json, metadata.json, metrics_overview.png}
├── transparency/default/{attributions.pt, CaptumImageVisualiser_0.png, metadata.json}
├── robustness/
│   ├── pgd/{robustness_data.pt, ImagePairVisualiser_0.png, metadata.json}
│   └── fgsm/{robustness_data.pt, ImagePairVisualiser_0.png, metadata.json}
└── reports/{report.html, report.zip, _assets/…}
```
