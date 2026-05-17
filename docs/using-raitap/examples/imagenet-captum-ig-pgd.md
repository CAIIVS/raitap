---
title: "ImageNet · Captum IG · Torchattacks PGD"
description: "ImageNet samples on vit_b_32, Captum Integrated Gradients for transparency, Torchattacks PGD for robustness, classification metrics, HTML report."
myst:
  html_meta:
    "description": "ImageNet samples on vit_b_32, Captum Integrated Gradients for transparency, Torchattacks PGD for robustness, classification metrics, HTML report."
---

# ImageNet · Captum IG · Torchattacks PGD

```{recipe}
:summary: ImageNet samples on `vit_b_32`, Captum Integrated Gradients for transparency, Torchattacks PGD for robustness, classification metrics, HTML report.

:yaml:
defaults:
  - raitap_schema
  - _self_
  - reporting: html
  - metrics: classification

hardware: gpu
experiment_name: example

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

:python:
from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, LabelsConfig
from raitap.metrics import Task, classification
from raitap.models import ModelConfig
from raitap.reporting import html
from raitap.robustness import image_pair, torchattacks
from raitap.transparency import captum, captum_image

cfg = AppConfig(
    hardware=Hardware.gpu,
    experiment_name="example",
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
    metrics=classification(task=Task.multiclass),
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
    },
    reporting=html(filename="report"),
)
outputs = run(cfg, auto_install=True)

:output:
outputs/<date>/<time>/
├── metrics/{metrics.json, artifacts.json, metadata.json, metrics_overview.png}
├── transparency/default/{attributions.pt, CaptumImageVisualiser_0.png, metadata.json}
├── robustness/pgd/{robustness_data.pt, ImagePairVisualiser_0.png, metadata.json}
└── reports/{report.html, report.zip, _assets/…}
```
