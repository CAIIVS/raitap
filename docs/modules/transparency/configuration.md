---
title: "Configuration"
description: "Inside the transparency key, you can configure one or more explainers. See {ref}modules-transparency-configuration-examples for the config shape."
myst:
  html_meta:
    "description": "Inside the transparency key, you can configure one or more explainers. See {ref}modules-transparency-configuration-examples for the config shape."
---

```{config-page}
:intro: This page describes how to configure the transparency module that
  computes and visualises attributions.

  Inside the `transparency` key, you can configure one or more explainers. See {ref}`modules-transparency-configuration-examples` for the config shape.

  See {doc}`frameworks-and-libraries` for the backend behaviour behind
  `_target_`, `algorithm`, and visualiser compatibility.

  RAITAP has two separate batch-size controls because predictions and
  explanations are different workload stages:

  - `data.forward_batch_size` controls the model forward pass used for
    predictions, metrics, report sample selection, and `call.target: auto_pred`.
  - `transparency.<explainer>.raitap.batch_size` controls attribution
    computation for that explainer. Expensive methods such as SHAP or Captum
    Occlusion often need a much smaller attribution batch size than the
    prediction batch size.

:option: _target_
:allowed: "CaptumExplainer", "ShapExplainer"
:default: null
:description: Hydra target for the explainer class.

:option: algorithm
:allowed: See {doc}`frameworks-and-libraries`
:default: null
:description: Name of the underlying explainability algorithm to use. The exact class is resolved by the selected explainer backend.

:option: constructor
:allowed: dict
:default: null
:description: Keyword arguments passed when constructing the explainer or
  underlying library object.

:option: call
:allowed: dict
:default: null
:description: Keyword arguments passed verbatim to the underlying library when
  computing attributions. Any nested dict with a `source` key is treated as a
  runtime data source. For a baseline / reference input prefer `raitap.baseline`
  (library-agnostic name) over the raw library kwarg here.

:option: raitap
:allowed: dict
:default: null
:description: RAITAP-owned runtime options such as batching, progress display,
  and sample-name metadata. These keys are not forwarded to the underlying
  explainability library.

:option: raitap.batch_size
:allowed: int
:default: None
:description: Batch size for computing attributions. If not specified, the explainer
  will compute attributions in a single pass. This controls only
  the explainer attribution stage. It does not control the initial prediction
  forward pass used for metrics, report sample ranking, or `auto_pred` targets;
  configure that with `data.forward_batch_size`.

:option: raitap.show_progress
:allowed: bool
:default: True
:description: Whether to show a progress bar when computing attributions.
  `raitap.max_batch_size` has been removed. Use `raitap.batch_size` instead.

:option: raitap.progress_desc
:allowed: str
:default: null
:description: Description of the progress bar.

:option: raitap.sample_names
:allowed: list[str]
:default: null
:description: Optional per-sample names for downstream visualisers. This can be
  injected at runtime from the data pipeline. If runtime sample names
  are provided, they take precedence over `raitap.sample_names` from config.
  The list length must equal the number of input samples `N`; a mismatch
  raises `raitap.utils.errors.SampleNamesLengthError` at factory entry.
  Omit `sample_names` to fall back to auto-derived sample ids from the
  data loader.

:option: raitap.show_sample_names
:allowed: bool
:default: False
:description: Default toggle for showing sample names in visualiser titles. Set
  the explainer-level default here under `raitap:`. If a specific visualiser
  needs different behaviour, override it with
  `visualisers[].call.show_sample_names`.

:option: raitap.input_metadata
:allowed: dict
:default: null
:description: Input modality + layout hints used by output-space inference
  and visualiser selection. Keys: `kind` (one of `image`, `tabular`, `text`,
  `time_series`), `layout` (`NCHW`, `(B,F)`, `(B,T,C)`, `TOKENS`),
  `feature_names`. Either `kind` or `layout` alone is enough to disambiguate
  the output space. The full run pipeline auto-infers `input_metadata` from
  `data.source` for image and tabular layouts, so most users won't need to
  set this. Direct callers of `infer_output_space` must pass it explicitly —
  otherwise the helper raises `ValueError`. This per-explainer metadata is
  scoped to output-space/visualiser semantics; backend input reshape is
  controlled by `data.input_metadata.shape` instead — see
  {doc}`../data/configuration`.

:option: raitap.baseline
:allowed: dict | tensor
:default: null
:description: Library-agnostic baseline / reference input for attribution methods
  that take one (Captum `baselines`, SHAP `background_data`). Setting it on an explainer that
  takes no baseline (e.g. Saliency) raises `raitap.utils.errors.RaitapError`. A
  single-reference method (e.g. Integrated Gradients) warns if given a multi-sample
  baseline (`n_samples > 1`) — that shape only works for a sample-set method like
  SHAP; use `n_samples: 1` for a broadcast reference. Omit it to fall back to the
  method's implicit default (zeros for IG, the input batch for SHAP).

:option: visualisers
:allowed: list[dict]
:default: []
:description: Visualiser definitions. Each entry must include at least
  `_target_`. Each visualiser can also define its own `constructor` and `call`
  blocks. Use `visualisers[].call.show_sample_names` for per-visualiser sample
  name overrides; use `raitap.show_sample_names` for the shared explainer-level
  default.

:yaml:
transparency:
  my_first_explainer:
    _target_: "CaptumExplainer"
    algorithm: "IntegratedGradients"
    call:
      target: 0
    raitap:
      baseline:                
        source: "./data/baseline"
      input_metadata:
        kind: image
        layout: NCHW
    visualisers:
      - _target_: "CaptumImageVisualiser"
        call:
          max_samples: 1
  my_second_explainer:
    _target_: "ShapExplainer"
    algorithm: "KernelExplainer"
    call:
      target: 0
    raitap:
      baseline:
        source: "./data/background"
        n_samples: 32
      batch_size: 1
      input_metadata:
        kind: tabular
        layout: "(B,F)"
        feature_names: [age, income, score]
    visualisers:
      - _target_: "ShapBarVisualiser"

:cli: transparency.captum_ig.algorithm=GradientShap

:python:
from raitap.transparency import captum, captum_image, shap, shap_bar

transparency = {
    "my_first_explainer": captum(
        algorithm="IntegratedGradients",
        call={"target": 0},
        raitap={
            "baseline": {"source": "./data/baseline"},  # routed to Captum's `baselines`
            "input_metadata": {"kind": "image", "layout": "NCHW"},
        },
        visualisers=[captum_image(call={"max_samples": 1})],
    ),
    "my_second_explainer": shap(
        algorithm="KernelExplainer",
        call={"target": 0},
        raitap={
            "baseline": {"source": "./data/background", "n_samples": 32},
            "batch_size": 1,
            "input_metadata": {
                "kind": "tabular",
                "layout": "(B,F)",
                "feature_names": ["age", "income", "score"],
            },
        },
        visualisers=[shap_bar()],
    ),
}
```

