---
title: "Configure SHAP Background Data"
description: "Run SHAP explainers with explicit background data and controlled batching."
---

SHAP explainers are usually the first place where assessment jobs become memory-sensitive. This guide shows the RAITAP-native way to pass background data and constrain batch sizes without rewriting application code.

## Problem

`GradientExplainer`, `DeepExplainer`, and `KernelExplainer` often need background data, and large input batches can make them slow or memory-heavy.

## Solution

Declare `background_data` inside `call:` and place runtime-only options under `raitap:`. RAITAP will load the background source at runtime and batch the input batch before calling the explainer.

<Steps>
<Step>
### Install SHAP-compatible extras
```bash
uv add "raitap[shap,reporting,torch-cpu]"
```
</Step>
<Step>
### Create a SHAP-oriented config
```yaml
defaults:
  - _self_
  - model: resnet50
  - data: imagenet_samples
  - reporting: disabled

experiment_name: shap-gradient
hardware: cpu

transparency:
  shap_gradient:
    _target_: ShapExplainer
    algorithm: GradientExplainer
    constructor:
      local_smoothing: 0.0
    call:
      target: 0
      nsamples: 10
      background_data:
        source: imagenet_samples
        n_samples: 2
    raitap:
      batch_size: 1
      progress_desc: "SHAP batches"
      input_metadata:
        kind: image
        layout: NCHW
    visualisers:
      - _target_: ShapImageVisualiser
        constructor:
          max_samples: 1
```
</Step>
<Step>
### Run the assessment
```bash
uv run raitap --config-name assessment
```
</Step>
</Steps>

What happens internally:

- `_resolve_call_data_sources()` in `src/raitap/transparency/factory.py` turns the `background_data` mapping into a tensor by calling `load_tensor_from_source()`.
- `AttributionOnlyExplainer._compute_with_optional_batches()` in `src/raitap/transparency/explainers/base_explainer.py` slices the input batch according to `raitap.batch_size`.
- `ShapExplainer.compute_attributions()` in `src/raitap/transparency/explainers/shap_explainer.py` uses that background tensor when constructing the SHAP explainer.

If you omit `background_data`, `ShapExplainer` falls back to using the current input batch for algorithms that require background data. That is convenient for demos, but it weakens the interpretability story for real experiments. A dedicated background dataset is usually the better choice.

For direct API usage, the same idea works without a full run:

```python
from raitap.data import load_tensor_from_source

background = load_tensor_from_source("imagenet_samples", n_samples=2)
```

This guide is the cleanest path when you need more control than the demo presets provide.
