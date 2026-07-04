---
title: "Object detection · Faster R-CNN · DetectionMetrics"
description: "Faster R-CNN (COCO) on dashcam frames, DetectionMetrics for mAP, Captum Integrated Gradients per detected box, DetectionImageVisualiser, HTML report."
myst:
  html_meta:
    "description": "Faster R-CNN (COCO) on dashcam frames, DetectionMetrics for mAP, Captum Integrated Gradients per detected box, DetectionImageVisualiser, HTML report."
---

# Object detection · Faster R-CNN · DetectionMetrics

```{recipe}
:summary: Faster R-CNN (COCO) on the bundled `UdacitySelfDriving` dashcam frames. `TorchBackend` auto-detects the `GeneralizedRCNN` head and routes the pipeline to the detection task family — `DetectionMetrics` (mean Average Precision via torchmetrics), per-box Integrated Gradients, and the box-aware `DetectionImageVisualiser`.

:yaml:
defaults:
  - raitap_schema
  - reporting: html
  - metrics: detection
  - data/labels: detection_json
  - _self_

hardware: cpu
experiment_name: detection-fasterrcnn

model:
  source: fasterrcnn_resnet50_fpn_v2
  # Optional: torchvision detectors auto-infer this. Set it for custom models.
  task_kind: detection

data:
  name: udacity-dashcam-demo
  source: UdacitySelfDriving
  forward_batch_size: 1
  labels:
    # JSON list-of-records. Each record: {sample_id, boxes: [[x1,y1,x2,y2], ...], labels: [coco_class_id, ...]}.
    # Coordinates are absolute pixels in xyxy format. See `docs/modules/data/configuration.md`.
    source: ./labels/udacity-boxes.json

metrics:
  # `metrics: detection` selects DetectionMetrics; overrides go below.
  class_metrics: true
  iou:
    thresholds: [0.5, 0.75]

transparency:
  # One per-box Integrated Gradients run. `call.target` must be 0 — the
  # ScalarDetectionWrapper exposes a single scalar channel per box, so
  # `auto_pred` is rejected. `max_boxes` caps the K-loop for CPU runs.
  detection_ig:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    call:
      target: 0
      n_steps: 8
      internal_batch_size: 1
    raitap:
      batch_size: 1
      detection:
        score_threshold: 0.5
        max_boxes: 3
        iou_threshold: 0.5
    visualisers:
      - _target_: DetectionImageVisualiser

:python:
from raitap import AppConfig, Hardware, run
from raitap.data import DataConfig, DetectionJsonLabelsConfig
from raitap.metrics import detection
from raitap.models import ModelConfig
from raitap.reporting import html
from raitap.transparency import captum, detection_image

cfg = AppConfig(
    hardware=Hardware.cpu,
    experiment_name="detection-fasterrcnn",
    # ``task_kind`` is optional for torchvision detectors (auto-inferred); set
    # it for custom models the inference can't recognise.
    model=ModelConfig(source="fasterrcnn_resnet50_fpn_v2", task_kind="detection"),
    data=DataConfig(
        name="udacity-dashcam-demo",
        source="UdacitySelfDriving",
        forward_batch_size=1,
        labels=DetectionJsonLabelsConfig(
            source="./labels/udacity-boxes.json",
        ),
    ),
    metrics=detection(
        class_metrics=True,
        iou={"thresholds": [0.5, 0.75]},
    ),
    transparency={
        "detection_ig": captum(
            algorithm="IntegratedGradients",
            call={"target": 0, "n_steps": 8, "internal_batch_size": 1},
            raitap={
                "batch_size": 1,
                "detection": {
                    "score_threshold": 0.5,
                    "max_boxes": 3,
                    "iou_threshold": 0.5,
                },
            },
            visualisers=[detection_image()],
        ),
    },
    reporting=html(filename="report"),
)
outputs = run(cfg, auto_install_deps=True)

:output:
outputs/<date>/<time>/
├── metrics/{metrics.json, artifacts.json, metadata.json, metrics_overview.png}
├── transparency/detection_ig/{attributions.pt, DetectionImageVisualiser_*.png, metadata.json}
└── reports/{report.html, report.zip, _assets/…}
```

## Labels file

`data.labels` selects the `detection_json` variant, whose `source` points at a
JSON list-of-records. Each record carries one
sample's ground-truth boxes (absolute pixels, `xyxy`) and COCO class ids
(class `3` = car):

```json
[
  {
    "sample_id": "straight_lines1.jpg",
    "boxes": [[659.7, 418.7, 676.6, 432.0], [676.4, 419.0, 687.2, 430.6]],
    "labels": [3, 3]
  }
]
```

`sample_id` matches the filename inside the `UdacitySelfDriving` sample
directory. See {doc}`../../modules/data/configuration` for the full schema.

## Notes

- `model.source: fasterrcnn_resnet50_fpn_v2` resolves to the torchvision
  builder name; `TorchBackend` loads the default COCO-pretrained weights and
  sets `task_kind = detection`, which steers the pipeline into the per-box
  explain phase and accepts `DetectionImageVisualiser` (rejected for
  classification runs).
- `transparency.<run>.raitap.detection` is the per-box K-loop budget:
  `score_threshold` drops low-confidence detections, `max_boxes` caps the loop
  for CPU runs, `iou_threshold` deduplicates overlapping boxes before
  attributing.
- `metrics: detection` selects the `DetectionMetrics` adapter (mean Average
  Precision via torchmetrics + `faster_coco_eval`). Knob reference lives at
  {doc}`../../modules/metrics/configuration`.
