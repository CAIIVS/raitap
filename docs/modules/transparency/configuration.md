```{config-page}
:intro: This page describes how to configure the transparency module that
  computes and visualises attributions.

  The top-level `transparency` section is a mapping of named explainers. Each
  entry has the same shape and is resolved independently by the pipeline.

  The shipped presets usually replace the default `default` entry with names
  such as `captum_ig` or `captum_saliency`.

  See [Frameworks and libraries](frameworks-and-libraries.md) for the backend
  behavior behind `_target_`, `algorithm`, and visualiser compatibility. See
  [Transparency Configuration](../../reference/config/transparency.md) for
  longer examples.

:option: _target_
:allowed: string
:default: "CaptumExplainer"
:description: Hydra target for the explainer class. This is commonly overridden
  by the selected transparency config group.

:option: algorithm
:allowed: string
:default: "IntegratedGradients"
:description: Name of the underlying explainability algorithm to use. The exact
  class is resolved by the selected explainer backend.

:option: constructor
:allowed: dict
:default: {}
:description: Keyword arguments passed when constructing the explainer or
  underlying library object.

:option: call
:allowed: dict
:default: {}
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
:default: "Computing attributions"
:description: Description of the progress bar.

:option: visualisers
:allowed: list[dict]
:default: [{"_target_": "CaptumImageVisualiser"}]
:description: Visualiser definitions. Each entry must include at least
  `_target_`. Each visualiser can also define its own `constructor` and `call`
  blocks.

:yaml:
transparency:
  captum_ig:
    _target_: "CaptumExplainer"
    algorithm: "IntegratedGradients"
    constructor: {}
    call:
      target: 0
    visualisers:
      - _target_: "CaptumImageVisualiser"
        constructor: {}
        call: {}

:cli: transparency.captum_ig.algorithm=GradientShap
```
