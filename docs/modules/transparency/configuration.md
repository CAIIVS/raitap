```{config-page}
:intro: This page describes how to configure the transparency module that
  computes and visualises attributions.

  Inside the `transparency` key, you can configure one or more explainers. See {ref}`modules-transparency-configuration-yaml-example` for the config shape.

  See {doc}`frameworks-and-libraries` for the backend behaviour behind
  `_target_`, `algorithm`, and visualiser compatibility.

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
:description: Keyword arguments passed when computing attributions. Any nested
  dict with a `source` key is treated as a runtime data source.

:option: call.batch_size
:allowed: int
:default: None
:description: Batch size for computing attributions. If not specified, the explainer
  will compute attributions in a single pass.

:option: call.max_batch_size
:allowed: int
:default: None
:description: Maximum batch size for computing attributions. If not specified, the explainer
  will compute attributions in a single pass.

:option: call.show_progress
:allowed: bool
:default: True
:description: Whether to show a progress bar when computing attributions.

:option: call.progress_desc
:allowed: str
:default: null
:description: Description of the progress bar.

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
      background_data:
        source: "./data/background"
        n_samples: 32
    visualisers:
      - _target_: "ShapImageVisualiser"

:cli: transparency.captum_ig.algorithm=GradientShap
```
