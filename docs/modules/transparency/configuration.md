```{config-page}
:intro: This page describes how to configure the transparency module that
  computes and visualises attributions.

  The top-level `transparency` section is a mapping. By default, it contains a
  single entry named `default`.

:option: _target_
:allowed: string
:default: "CaptumExplainer"
:description: Hydra target for the explainer class. This is commonly overridden
  by the selected transparency config group.

:option: algorithm
:allowed: string
:default: "IntegratedGradients"
:description: Name of the underlying explainability algorithm to use.

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

:option: visualisers
:allowed: list[dict]
:default: [{"_target_": "CaptumImageVisualiser"}]
:description: Visualiser definitions. Each entry must include at least
  `_target_`.

:yaml:
transparency:
  default:
    _target_: "CaptumExplainer"
    algorithm: "IntegratedGradients"
    constructor: {}
    call: {}
    visualisers:
      - _target_: "CaptumImageVisualiser"

:cli: transparency.default.algorithm=IntegratedGradients
```
