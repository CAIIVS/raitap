---
title: "Use Custom Data And Labels"
description: "Load your own image or tabular dataset and align labels safely."
---

This guide covers the most common production scenario: using your own local data and a separate labels file instead of the built-in sample sets.

## Problem

You have a dataset on disk and need RAITAP to load it, infer the input modality, and match labels reliably.

## Solution

Use `data.source` for the dataset root and `data.labels.*` for label-file alignment. Prefer an explicit ID column whenever your dataset contains nested directories or repeated filenames.

<Steps>
<Step>
### Create an assessment config
```yaml
defaults:
  - _self_
  - model: resnet50
  - metrics: classification
  - reporting: disabled

experiment_name: custom-images
hardware: cpu

data:
  name: chest-xray
  source: ./data/images
  labels:
    source: ./data/labels.csv
    id_column: image
    column: label
    id_strategy: relative_path

transparency:
  captum_ig:
    _target_: CaptumExplainer
    algorithm: IntegratedGradients
    call:
      target: auto_pred
    visualisers:
      - _target_: CaptumImageVisualiser
```
</Step>
<Step>
### Preview the resolved config
```bash
uv run raitap --config-name assessment --cfg job
```
</Step>
<Step>
### Execute the run
```bash
uv run raitap --config-name assessment
```
</Step>
</Steps>

Complete label file example:

```csv
image,label
NORMAL/IM-0001.jpeg,0
PNEUMONIA/IM-0002.jpeg,1
```

The important part is `id_strategy: relative_path`. In `src/raitap/data/data.py`, `_align_labels_to_samples()` normalizes both sample IDs and label IDs before matching them. With a nested dataset, stem-only matching can collapse two different files into the same ID.

For tabular data, the same pattern works with a feature matrix:

```yaml
data:
  name: risk-features
  source: ./data/features.parquet
  labels:
    source: ./data/targets.csv
    column: target
```

Because `infer_data_input_metadata()` in `src/raitap/data/metadata.py` recognizes `.csv`, `.tsv`, and `.parquet`, the transparency layer can infer tabular semantics automatically for the full pipeline.

If label alignment fails, RAITAP warns and falls back to predictions as targets through `resolve_metric_targets()` in `src/raitap/metrics/inputs.py`. That keeps the run alive, but it also means the metrics stop being a real evaluation. Treat that warning as a data-quality issue, not as a harmless fallback.
