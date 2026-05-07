---
title: "Getting Started"
description: "Install RAITAP, run the default assessment, and understand what the library gives you."
---

RAITAP is a Python library for running responsible-AI assessments on models with a single pipeline that combines model loading, data loading, explainability, metrics, tracking, and PDF reporting.

## The Problem

- Explainability workflows usually depend on framework-specific code, so switching between Captum and SHAP means rewriting both configuration and output handling.
- Model assessment code often breaks when teams mix PyTorch checkpoints, TorchScript artifacts, and ONNX exports in the same workflow.
- MLOps runs produce many artifacts, but they are rarely packaged into one repeatable pipeline with tracking metadata and human-readable reports.
- Dataset labels, sample IDs, and visualization outputs are easy to misalign, which makes comparisons and audits unreliable.

## The Solution

RAITAP centralizes those concerns in `raitap.run.pipeline.run`, which instantiates a `Model`, loads a `Data` batch, computes metrics, runs one or more transparency explainers, renders visualisations, optionally logs everything to MLflow, and optionally produces a PDF report. The orchestration lives in `src/raitap/run/pipeline.py`, while Hydra-backed configs in `src/raitap/configs/` keep the workflow declarative.

```python
from hydra import compose, initialize_config_dir
from pathlib import Path

from raitap.run import run

config_dir = Path("src/raitap/configs").resolve()

with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
    cfg = compose(
        config_name="config",
        overrides=[
            "hardware=cpu",
            "model=resnet50",
            "data=imagenet_samples",
            "transparency=demo",
            "reporting=disabled",
        ],
    )

outputs = run(cfg)
print(len(outputs.explanations))
print(outputs.forward_output.shape)
```

The returned `RunOutputs` object gives you structured access to the assessment result instead of forcing you to scrape files from disk.

## Installation

<Callout type="info">RAITAP is published as a Python package, not an npm package. The tabs below are included to match the documentation shell used by this site; every tab installs the same Python dependency set.</Callout>

" "bun"]}>
<Tab value="npm">
```bash
# RAITAP is Python-only.
uv add "raitap[captum,reporting,torch-cpu]"
```
</Tab>
<Tab value="pnpm">
```bash
# RAITAP is Python-only.
uv add "raitap[captum,reporting,torch-cpu]"
```
</Tab>
<Tab value="yarn">
```bash
# RAITAP is Python-only.
uv add "raitap[captum,reporting,torch-cpu]"
```
</Tab>
<Tab value="bun">
```bash
# RAITAP is Python-only.
uv add "raitap[captum,reporting,torch-cpu]"
```
</Tab>
</Tabs>

The core package requires Python `>=3.13,<3.14` as declared in `pyproject.toml`. Optional extras activate runtime families such as `torch-cpu`, `onnx-cpu`, `captum`, `shap`, `metrics`, `mlflow`, and `reporting`.

## Quick Start

The smallest working run is the default CLI entry point defined by `pyproject.toml` as `raitap = "raitap.run.__main__:main"`.

```bash
uv run raitap hardware=cpu reporting=disabled
```

Expected output:

```text
============================================================
RAITAP Assessment
============================================================
Experiment: demo
Model: vit_b_32
Dataset: isic2018
Hardware: CPU
Explainers: ['captum_ig', 'captum_saliency']
Metrics: on
Output: ./outputs/<date>/<time>
```

And inside the run directory you should see a structure close to this:

```text
outputs/<date>/<time>/
  metrics/
    metrics.json
    artifacts.json
    metadata.json
  transparency/
    captum_ig/
      attributions.pt
      metadata.json
      CaptumImageVisualiser_0.png
  reports/   # only when reporting is enabled
```

## Key Features

- One pipeline for PyTorch and ONNX model assets through `raitap.models.Model`.
- Hydra-native configuration presets in `src/raitap/configs/` with CLI override support.
- Framework-agnostic transparency orchestration with Captum and SHAP wrappers.
- Structured outputs for metrics, explanations, visualisations, tracking, and reports.
- Optional MLflow logging and optional PDF report generation.
- Source-aware data loading for local files, directories, URLs, and built-in demo datasets.

<Cards>
  <Card title="Architecture" href="/docs/architecture">See how the pipeline composes models, data, metrics, explainers, tracking, and reporting.</Card>
  <Card title="Core Concepts" href="/docs/hydra-configuration">Learn the config model, runtime semantics, and artifact lifecycle.</Card>
  <Card title="API Reference" href="/docs/api-reference/run">Jump to the exported classes, functions, and public dataclasses.</Card>
</Cards>
