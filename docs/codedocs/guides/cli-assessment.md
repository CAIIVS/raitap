---
title: "Run A CLI Assessment"
description: "Compose a practical RAITAP assessment from presets and CLI overrides."
---

This guide shows the fastest realistic path from installation to a local assessment run. The goal is a reproducible CPU-based job that loads a built-in model, uses demo image data, computes classification metrics, renders Captum explanations, and writes artifacts locally.

## Problem

You want to verify that RAITAP works in your environment before wiring it into a bigger project.

## Solution

Use the built-in Hydra presets and only override the parts that affect reproducibility: hardware and reporting mode.

<Steps>
<Step>
### Install the runtime extras
```bash
uv add "raitap[captum,metrics,reporting,torch-cpu]"
```
</Step>
<Step>
### Run the assessment
```bash
uv run raitap \
  hardware=cpu \
  model=resnet50 \
  data=imagenet_samples \
  transparency=demo
```
</Step>
<Step>
### Inspect the artifacts
```bash
find outputs -maxdepth 4 -type f | sort
```
</Step>
</Steps>

The `demo` transparency preset from `src/raitap/configs/transparency/demo.yaml` runs two explainers: `captum_ig` and `captum_saliency`. Because `reporting=pdf` is part of the default top-level config, you also get a report unless you explicitly disable it.

Complete example config override set:

```bash
uv run raitap \
  hardware=cpu \
  model=resnet50 \
  data=imagenet_samples \
  transparency=demo \
  metrics=classification \
  experiment_name=cpu-demo
```

Expected results:

- `metrics/metrics.json` with scalar accuracy, precision, recall, and F1 values
- one transparency subdirectory per explainer
- PNG visualisations from `CaptumImageVisualiser`
- a PDF report under `reports/report.pdf`

If you want to preview the resolved config first, Hydra supports:

```bash
uv run raitap --cfg job hardware=cpu model=resnet50 data=imagenet_samples transparency=demo
```

This guide maps directly onto the main pipeline in `src/raitap/run/pipeline.py`, so it is also the best smoke test for a fresh installation.

If the run fails on the first attempt, check the optional extras first. A successful CLI smoke test usually proves that the core dependencies, Hydra config registration, model loading path, data loader, and at least one transparency backend are all wired correctly in the current environment.
