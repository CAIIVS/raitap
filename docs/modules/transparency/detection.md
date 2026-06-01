---
title: "Detection knobs"
description: "Per-box detection explanation configuration: score_threshold, max_boxes, iou_threshold."
myst:
  html_meta:
    "description": "Per-box detection explanation configuration: score_threshold, max_boxes, iou_threshold."
---

# Detection knobs

```{config-page}
:slug: detection
:intro: For backends whose `task_kind == detection` (e.g. torchvision
  Faster R-CNN / RetinaNet / SSD), the pipeline switches to a per-box
  explanation loop that emits one `ExplanationResult` per detected box.
  Knobs live under the explainer's `raitap.detection` block; defaults
  apply when omitted.

  The explainer's `call.target` MUST be `0` for detection runs (the
  `ScalarDetectionWrapper` exposes one scalar channel; `target=0` selects
  it). `call.target: auto_pred` is rejected with a `RaitapError` because
  argmax over a 1-channel output always returns 0, which would mask real
  config errors.

:option: raitap.detection.score_threshold
:allowed: float
:default: 0.5
:description: Drop detections whose score is strictly below this before
  selecting boxes. Applies per-sample; samples with no detections above
  the threshold emit a warning and skip the K-loop.

:option: raitap.detection.max_boxes
:allowed: int
:default: 5
:description: Cap K per sample after threshold filtering, by score
  descending. Smaller values keep CI / demo runs fast; larger values give
  more attribution coverage at the cost of one extra explainer call per
  box.

:option: raitap.detection.iou_threshold
:allowed: float
:default: 0.5
:description: IoU threshold used by the `reference_match` target.
  Perturbed predictions must have at least this IoU with the original
  box (and a matching label) to count toward the explained scalar; below
  this the scalar smoothly degrades, preserving autograd for gradient
  explainers.

:yaml:
transparency:
  my_ig_explainer:
    _target_: "CaptumExplainer"
    algorithm: "IntegratedGradients"
    call:
      target: 0
    raitap:
      detection:
        score_threshold: 0.5
        max_boxes: 5
        iou_threshold: 0.5
    visualisers:
      - _target_: "DetectionImageVisualiser"

:cli: transparency.my_ig_explainer.raitap.detection.max_boxes=3

:python:
from raitap.transparency import captum, detection_image

transparency = {
    "my_ig_explainer": captum(
        algorithm="IntegratedGradients",
        call={"target": 0},
        raitap={
            "detection": {
                "score_threshold": 0.5,
                "max_boxes": 5,
                "iou_threshold": 0.5,
            }
        },
        visualisers=[detection_image()],
    ),
}
```

## Box labels and ground truth

Each explained box is labelled in the report headings, figure titles, and
thumbnail overlay with its **predicted class name** and, when ground truth is
available, the **true label** of the closest real object.

**Predicted class names.** A box reads `kite` instead of `class 38` whenever a
class-name source is available:

- For pretrained torchvision detectors, the model's bundled category names are
  used automatically — no configuration needed.
- For your own model, set `model.class_names` to the id-ordered list of names
  (index 0 first). This takes precedence over any bundled names.
- With neither, boxes fall back to the numeric form `class <id>`.

**True labels.** When you configure detection ground truth
(`data.labels.source` with `data.labels.kind: detection`), each box is matched
to the ground-truth object it overlaps most and shows that object's label plus
the overlap as `gt: <name> (IoU <value>)`. The match is by overlap alone, so a
disagreement is visible directly — e.g. `pred: dog 0.92 | gt: cat, IoU=0.71`.
A box that overlaps no labelled object reads `gt: no match`. IoU (intersection
over union) ranges 0–1; the match uses the same `raitap.detection.iou_threshold`
as the explanation, and a higher value means a tighter overlap.

With no ground truth configured, no `gt:` line is shown.
