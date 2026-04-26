```{config-page}
:intro: This page describes how to configure the transparency module that
  computes and visualises attributions.

  Inside the `transparency` key, you can configure one or more explainers. See {ref}`modules-transparency-configuration-yaml-example` for the config shape.

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

:option: raitap.show_sample_names
:allowed: bool
:default: False
:description: Default toggle for showing sample names in visualiser titles. Set
  the explainer-level default here under `raitap:`. If a specific visualiser
  needs different behaviour, override it with
  `visualisers[].call.show_sample_names`.

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
    visualisers:
      - _target_: "ShapImageVisualiser"

:cli: transparency.captum_ig.algorithm=GradientShap
```
