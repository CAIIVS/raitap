# Faster R-CNN / Udacity dashcam

COCO-pretrained `fasterrcnn_resnet50_fpn_v2` (torchvision) on four 1280x720
Udacity CarND dashcam frames, with per-box Integrated Gradients attribution
and detection mAP. Tuned to run on CPU; finishes in ~30s on a decent laptop.

## Before you run

1. **Labels.** The pipeline needs `labels/udacity-boxes.json` to score
   predictions. The file is not auto-generated — see
   [`labelling-data.md`](./labelling-data.md) for the JSON format, the COCO
   class indices that matter for dashcam scenes, and the bundled
   candidate-generation helper (`scripts/generate_candidate_labels.py`)
   that does most of the work.

2. **Data.** The four JPEGs download automatically on first run into
   `~/.cache/raitap/UdacitySelfDriving/` (no manual fetch needed).

3. **Model.** Resolved by torchvision builder name — `weights="DEFAULT"`
   pulls the COCO-pretrained checkpoint on first import.

## Run it

From the repo root:

```bash
# Linux / macOS
contributor-configs/fasterrcnn-udacity/run.sh
```

```powershell
# Windows
contributor-configs\fasterrcnn-udacity\run.ps1
```

Both wrap `uv run raitap` with the right extras (`torch-cpu`, `captum`,
`metrics`, `reporting`) and pass `--acknowledge-preprocessing-off` —
torchvision detection models do their own internal resize and normalisation,
so RAITAP-level preprocessing is intentionally absent (adding one would
break box coordinate alignment).

## Outputs

Hydra writes the run directory under `outputs/<date>/<time>/`. After a
successful run you get:

- `metrics/metrics.json` — DetectionMetrics summary (mAP / mAR over the
  four-image sample). Treat as a sanity check, not a benchmark.
- `transparency/detection_ig/` — one PNG per (sample, kept box) showing the
  cropped detection and its Integrated Gradients attribution heatmap. With
  `max_boxes: 3` and `score_threshold: 0.5` you get at most three boxes per
  frame.
- `reports/report.html` — single-page HTML report bundling the metrics
  table, the detection PNGs, and the resolved config.

## GPU

CPU is the default for safe CI. To run on GPU, either edit the YAML
(`hardware: gpu`) or override on the CLI:

```bash
contributor-configs/fasterrcnn-udacity/run.sh hardware=gpu
```

Same for `run.ps1`. Swap `--extra torch-cpu` for `--extra torch-cuda` (or
`--extra torch-intel`) in the script if you want the matching torch wheel
installed at sync time.
