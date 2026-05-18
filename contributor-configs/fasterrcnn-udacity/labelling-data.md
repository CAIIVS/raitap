uv run python -c "from raitap.data.samples import _load_sample; _load_sample('UdacitySelfDriving')"
  uv run --extra torch-intel python contributor-configs/fasterrcnn-udacity/scripts/generate_candidate_labels.py

# Labelling the Udacity dashcam samples

The `fasterrcnn-udacity` contributor config needs per-image ground-truth
detection labels so the metric pipeline (`DetectionMetrics` / mAP) has
something honest to score predictions against. This file documents how to
produce `labels/udacity-boxes.json`.

## What we're labelling

Four 1280×720 dashcam frames from
[`udacity/CarND-Advanced-Lane-Lines`](https://github.com/udacity/CarND-Advanced-Lane-Lines):

- `straight_lines1.jpg`
- `straight_lines2.jpg`
- `test1.jpg`
- `test2.jpg`

RAITAP caches them at `~/.cache/raitap/UdacitySelfDriving/` on first use.
Fetch on demand:

```bash
uv run python -c "from raitap.data.samples import _load_sample; _load_sample('UdacitySelfDriving')"
```

## Output format

A single JSON array. One record per image, ordered freely (RAITAP aligns by
`sample_id`).

```json
[
  {
    "sample_id": "straight_lines1.jpg",
    "boxes": [
      [x1, y1, x2, y2],
      [x1, y1, x2, y2]
    ],
    "labels": [3, 3]
  },
  {
    "sample_id": "straight_lines2.jpg",
    "boxes": [],
    "labels": []
  }
]
```

### Rules

| Field | Constraint |
|---|---|
| `sample_id` | Must match filename exactly. All four must be present. |
| `boxes` | xyxy in pixel space (top-left origin, PIL convention). `x1<x2`, `y1<y2`. Floats OK. |
| `labels` | COCO class indices (integers). `len(labels) == len(boxes)` per record. |
| Empty image | `"boxes": []` + `"labels": []` is valid — pass-through to metric computer. |

### COCO class indices (relevant for dashcam)

| ID | Class |
|---|---|
| 1 | person |
| 2 | bicycle |
| 3 | car |
| 4 | motorcycle |
| 6 | bus |
| 8 | truck |
| 10 | traffic light |
| 13 | stop sign |

Full COCO label list lives in `torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]`.

## Recommended workflow

### Option A — pure hand-labelling

1. Open each image in a viewer that reports pixel coords (Windows Photos,
   IrfanView, GIMP, online tool like [makesense.ai](https://www.makesense.ai)).
2. Note xyxy + class for each visible object you care about.
3. Write JSON manually.

### Option B — generate-then-edit (recommended, faster)

Run the bundled helper to get candidate labels from the same
`fasterrcnn_resnet50_fpn_v2` model the config ships with, then trim:

```bash
uv run --extra torch-intel python contributor-configs/fasterrcnn-udacity/scripts/generate_candidate_labels.py
```

Writes `labels/udacity-boxes.candidate.json` with all predictions above
score 0.5. Console output shows class name + score + box for each candidate.

Then:

1. Open `labels/udacity-boxes.candidate.json` in an editor.
2. Open each image side-by-side.
3. **Delete** rows for obvious false positives.
4. **Fix** wrong labels (e.g. truck mislabelled as bus).
5. **Add** any boxes the model missed but you can clearly see.
6. Round coords to 1 decimal place if you want (already done by the script).
7. Save as `labels/udacity-boxes.json`. The `.candidate.json` file is git-ignored — do not commit it.

### Option C — accept model output as-is (not recommended)

Just rename `labels/udacity-boxes.candidate.json` → `labels/udacity-boxes.json`. mAP will be near 1.0 because predictions match "labels" by construction. Pipeline runs but the metric tells you nothing about model quality — only that the code path works. Fine for smoke-testing; useless for actual evaluation.

## Why `labels/` and not `artifacts/`?

`contributor-configs/*/artifacts/` is git-ignored (it's the convention for
downloadable assets like model weights and image bundles that don't belong
in source control). Detection labels are small JSON, must ship with the
repo for CI to run the E2E pipeline, and are config-shaped — they live
under `labels/` instead. The `labels/*.candidate.json` glob is git-ignored
so helper-script output never lands in commits.

## Sanity checks before committing

- File parses as JSON: `uv run python -m json.tool labels/udacity-boxes.json > /dev/null`
- Four records, one per image
- Every `sample_id` matches an actual filename
- Every box satisfies `x1<x2` and `y1<y2`
- Coords within `[0, 1280]` × `[0, 720]`
- Every label is a valid COCO index (1–90, with gaps — check the categories list)

CI's detection E2E job loads this file and runs the full pipeline — broken
JSON or out-of-bounds boxes will surface there.
