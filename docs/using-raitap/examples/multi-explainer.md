---
title: "Multi-explainer · Captum IG + Saliency"
description: "Two transparency explainers under one run — Captum Integrated Gradients and Captum Saliency side by side, each producing its own attribution + visualisation row in the HTML report."
myst:
  html_meta:
    "description": "Two transparency explainers under one run — Captum Integrated Gradients and Captum Saliency side by side, each producing its own attribution + visualisation row in the HTML report."
---

# Multi-explainer · Captum IG + Saliency

```{recipe}
:summary: Two transparency explainers under one run — Captum Integrated Gradients and Captum Saliency side by side, each producing its own attribution + visualisation row in the HTML report.

:yaml:
defaults:
  - raitap_schema
  - reporting: html
  - metrics: multiclass_classification
  - data/labels: tabular
  - _self_

hardware: gpu
experiment_name: multi-explainer

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
  # Sample-frequency-weighted average instead of the macro default — more
  # meaningful on imbalanced datasets where rare classes shouldn't dominate.
  average: weighted

transparency:
  ig:
    use: captum
    algorithm: IntegratedGradients
    call:
      target: 0
    visualisers:
      - use: captum_image
  saliency:
    use: captum
    algorithm: Saliency
    call:
      target: 0
    visualisers:
      - use: captum_image

:python:
from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, TabularLabelsConfig
from raitap.metrics import multiclass_classification
from raitap.models import ModelConfig
from raitap.reporting import html
from raitap.transparency import captum, captum_image

cfg = AppConfig(
    hardware=Hardware.gpu,
    experiment_name="multi-explainer",
    model=ModelConfig(source="vit_b_32"),
    data=DataConfig(
        name="imagenet_samples",
        source="imagenet_samples",
        forward_batch_size=4,
        labels=TabularLabelsConfig(
            source="imagenet_samples",
            id_column="image",
            column="label",
        ),
    ),
    metrics=multiclass_classification(num_classes=1000, average="weighted"),
    transparency={
        "ig": captum(
            algorithm="IntegratedGradients",
            call={"target": 0},
            visualisers=[captum_image()],
        ),
        "saliency": captum(
            algorithm="Saliency",
            call={"target": 0},
            visualisers=[captum_image()],
        ),
    },
    reporting=html(filename="report"),
)
outputs = run(cfg, auto_install_deps=True)

:output:
outputs/<date>/<time>/
├── metrics/{metrics.json, artifacts.json, metadata.json, metrics_overview.png}
├── transparency/
│   ├── ig/{attributions.pt, CaptumImageVisualiser_0.png, metadata.json}
│   └── saliency/{attributions.pt, CaptumImageVisualiser_0.png, metadata.json}
└── reports/{report.html, report.zip, _assets/…}
```
