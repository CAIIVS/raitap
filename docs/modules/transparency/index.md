---
title: "Transparency"
description: "The transparency module configures the explainers and visualisers that generate attributions for your model predictions."
myst:
  html_meta:
    "description": "The transparency module configures the explainers and visualisers that generate attributions for your model predictions."
---

# Transparency

The transparency module configures the explainers and visualisers that generate
attributions for your model predictions.

Each `transparency` entry defines one named explainer, its algorithm, and the
visualisers that should render its outputs. The current implementation supports
Captum- and SHAP-based explainers behind the same config surface.

## Grading explanation quality (Quantus)

Add an `evaluation:` block under any explainer to grade its attributions with
[Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus)
metrics: faithfulness, complexity, robustness, localisation, randomisation, and
axiomatic checks. Grading runs as a post-step after the explainer produces its
attributions; scores land in `TransparencyOutput.evaluations`. See
{doc}`configuration` for the config shape and {doc}`output` for how scores and
skipped metrics are reported.

This needs the `quantus` extra: `uv sync --extra quantus`. It is not pulled in
by the `transparency` umbrella extra, so install it explicitly.

```{toctree}
:maxdepth: 1
:caption: Transparency module documentation

configuration
detection
frameworks-and-libraries
output
visualisers
```
