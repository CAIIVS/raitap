```{config-page}
:intro: This page describes how to configure the model that RAITAP will assess.

:option: source
:allowed: string, null
:default: null
:description: Path to a local model file, or a built-in model name such as
  `"resnet50"`. Refer to [Using your own model or built-in models](own-vs-built-in.md)
  for more details.

:yaml:
model:
  source: "myModel.pth"

:cli: model.source=resnet50
```
