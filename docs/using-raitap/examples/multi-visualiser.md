---
title: "Multi-visualiser · one explainer, two rendering styles"
description: "One Captum Integrated Gradients explainer with two visualisers — a blended heat-map and a plain heat-map of the same attributions. Demonstrates the visualisers: list on a single transparency run."
myst:
  html_meta:
    "description": "One Captum Integrated Gradients explainer with two visualisers — a blended heat-map and a plain heat-map of the same attributions. Demonstrates the visualisers: list on a single transparency run."
---

# Multi-visualiser · one explainer, two rendering styles

```{recipe}
:summary: One Captum Integrated Gradients explainer with two visualisers — a blended heat-map and a plain heat-map of the same attributions. Demonstrates the `visualisers:` list on a single transparency run.

:yaml:
defaults:
  - raitap_schema
  - reporting: html
  - metrics: multiclass_classification
  - data/labels: tabular
  - _self_

hardware: gpu
experiment_name: multi-visualiser

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
  # Drop a sentinel label from the metric — useful when labels.csv encodes
  # "unknown" / "background" as -1 and you want them excluded from accuracy.
  ignore_index: -1

transparency:
  default:
    use: captum
    algorithm: IntegratedGradients
    call:
      target: 0
    visualisers:
      - use: captum_image
        constructor:
          method: blended_heat_map
          sign: all
          title: Integrated gradients (blended)
      - use: captum_image
        constructor:
          method: heat_map
          sign: absolute_value
          title: Integrated gradients (absolute)

:python:
from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, TabularLabelsConfig
from raitap.metrics import multiclass_classification
from raitap.models import ModelConfig
from raitap.reporting import html
from raitap.transparency import captum, captum_image

cfg = AppConfig(
    hardware=Hardware.gpu,
    experiment_name="multi-visualiser",
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
    metrics=multiclass_classification(num_classes=1000, ignore_index=-1),
    transparency={
        "default": captum(
            algorithm="IntegratedGradients",
            call={"target": 0},
            visualisers=[
                captum_image(
                    method="blended_heat_map",
                    sign="all",
                    title="Integrated gradients (blended)",
                ),
                captum_image(
                    method="heat_map",
                    sign="absolute_value",
                    title="Integrated gradients (absolute)",
                ),
            ],
        ),
    },
    reporting=html(filename="report"),
)
outputs = run(cfg, auto_install_deps=True)

:output:
outputs/<date>/<time>/
├── metrics/{metrics.json, artifacts.json, metadata.json, metrics_overview.png}
├── transparency/default/{attributions.pt, CaptumImageVisualiser_0.png, CaptumImageVisualiser_1.png, metadata.json}
└── reports/{report.html, report.zip, _assets/…}
```
