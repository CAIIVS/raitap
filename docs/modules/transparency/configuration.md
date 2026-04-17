```{config-page}
:intro: This page describes how to configure the transparency module that
  computes and visualises attributions.

  Inside the `transparency` key, you can configure one or more explainers. See {ref}`modules-transparency-configuration-yaml-example` for the config shape.

  See {doc}`frameworks-and-libraries` for the backend behaviour behind
  `_target_`, `algorithm`, and visualiser compatibility.

:option: _target_
:allowed: "CaptumExplainer", "ShapExplainer", "AlibiExplainer"
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
  runtime data source.

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
  will compute attributions in a single pass. Currently implemented for Captum and
  SHAP explainers; Alibi ignores this key and emits a warning.

:option: raitap.show_progress
:allowed: bool
:default: True
:description: Whether to show a progress bar when computing attributions. Currently
  implemented for Captum and SHAP explainers; Alibi ignores this key and emits a warning.
  `raitap.max_batch_size` has been removed. Use `raitap.batch_size` instead.

:option: raitap.progress_desc
:allowed: str
:default: null
:description: Description of the progress bar. Currently implemented for Captum and
  SHAP explainers; Alibi ignores this key and emits a warning.

:option: raitap.sample_names
:allowed: list[str]
:default: null
:description: Optional per-sample names for downstream visualisers. This is
  usually injected at runtime from the data pipeline.

:option: raitap.show_sample_names
:allowed: bool
:default: False
:description: Default toggle for showing sample names in visualiser titles.

:option: visualisers
:allowed: list[dict]
:default: []
:description: Visualiser definitions. Each entry must include at least
  `_target_`. Each visualiser can also define its own `constructor` and `call`
  blocks.

:yaml:
transparency:
  my_first_explainer:
    _target_: "CaptumExplainer"
    algorithm: "IntegratedGradients"
    call:
      target: 0
    visualisers:
      - _target_: "CaptumImageVisualiser"
        call:
          max_samples: 1
  my_second_explainer:
    _target_: "ShapExplainer"
    algorithm: "GradientExplainer"
    constructor:
      local_smoothing: 0.0
    call:
      target: 0
      background_data:
        source: "./data/background"
        n_samples: 32
    raitap:
      batch_size: 1
      show_progress: true
    visualisers:
      - _target_: "ShapImageVisualiser"

:cli: transparency.captum_ig.algorithm=GradientShap
```
