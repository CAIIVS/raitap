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

Boxes are labelled in the report headings and on the thumbnail overlay; the
per-box attribution figures stay title-less (a title would only duplicate them).

**Predicted name.** A box reads `kite`, not `class 38`, when a name source is
available: torchvision detectors use their bundled categories automatically;
otherwise set `model.class_names` (id-ordered, index 0 first), which takes
precedence. With neither, boxes fall back to `class <id>`.

**True label.** Set `data.labels.source` to match
each box to the ground-truth object it overlaps most, shown as
`gt: <name> (IoU <value>)`. The match is by overlap alone, so disagreements
surface (`pred: dog 0.92 | gt: cat (IoU 0.71)`); a box overlapping no labelled
object reads `gt: no match`, and with no ground truth the `gt:` clause is
omitted. Matching reuses `raitap.detection.iou_threshold`.
