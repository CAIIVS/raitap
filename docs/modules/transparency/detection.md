---
title: "Detection knobs"
description: "Per-box detection explanation configuration: score_threshold, max_boxes, iou_threshold."
myst:
  html_meta:
    "description": "Per-box detection explanation configuration: score_threshold, max_boxes, iou_threshold."
---

# Detection knobs

For backends whose `task_kind == detection` (e.g. torchvision Faster R-CNN /
RetinaNet / SSD), the pipeline switches to a per-box explanation loop that
emits one `ExplanationResult` per detected box. Knobs live under the
explainer's `raitap.detection` block; defaults apply when omitted.

| Key | Default | Meaning |
|---|---|---|
| `score_threshold` | `0.5` | Drop detections whose score is strictly below this before selecting boxes. |
| `max_boxes` | `5` | Cap K per sample after threshold filtering, by score descending. |
| `iou_threshold` | `0.5` | IoU threshold used by the `reference_match` target — perturbed predictions need at least this IoU with the original box (and matching label) to count toward the explained scalar. |

The explainer's `call.target` MUST be `0` for detection runs (the
`ScalarDetectionWrapper` exposes one scalar channel; `target=0` selects it).
`call.target: auto_pred` is rejected with a `RaitapError` because argmax
over a 1-channel output always returns 0, which would mask real config
errors.

```yaml
transparency:
  my_ig_explainer:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    call:
      target: 0
    raitap:
      detection:
        score_threshold: 0.5
        max_boxes: 5
        iou_threshold: 0.5
    visualisers:
      - _target_: DetectionImageVisualiser
```
