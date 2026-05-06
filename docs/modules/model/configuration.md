```{config-page}
:intro: This page describes how to configure the model that RAITAP will assess.

:option: source
:allowed: string, null
:default: null
:description: Path to a local model file, or a built-in model name such as
  `"resnet50"`. Refer to {doc}`own-vs-built-in` for more details.

:option: arch
:allowed: string, null
:default: null
:description: torchvision architecture name (e.g. `"resnet18"`) used to
  instantiate the model when `source` points at a state-dict (a `.pt` /
  `.pth` file produced by `torch.save(model.state_dict(), path)`). Required
  alongside `num_classes` for state-dict loading; ignored for full pickled
  modules and TorchScript archives.

:option: num_classes
:allowed: int, null
:default: null
:description: Output classes used to instantiate the architecture before
  `load_state_dict`. Required together with `arch` for state-dict loading.

:option: pretrained
:allowed: bool
:default: false
:description: If `true`, construct the architecture with ImageNet pretrained
  weights before loading the state-dict (`weights="DEFAULT"`). Usually
  `false` since the state-dict already supplies the weights.

:yaml:
# Option A — full pickled nn.Module (deprecated, fragile across environments):
model:
  source: "myModel.pth"

# Option B — state_dict + arch (recommended):
model:
  source: "weights.pth"
  arch: "resnet18"
  num_classes: 2

# Option C — TorchScript archive (env-independent):
model:
  source: "scripted.pt"

:cli: model.source=resnet50
```
