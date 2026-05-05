# Models Module

This module loads machine-learning models into a uniform `ModelBackend`
that downstream RAITAP modules (transparency, metrics, reporting) consume
without caring about how the model was serialised.

It supports torchvision built-ins, ONNX archives, and three flavours of
PyTorch checkpoint.

## Loader behaviour for `.pt` / `.pth`

Order of attempts when `model.source` points at a `.pt` or `.pth` file:

1. **TorchScript** (`torch.jit.load`) — if the file is a TorchScript
   archive, return the `ScriptModule` as-is. No extra config keys needed.
2. **state_dict** — if `torch.load` returns a `dict`, RAITAP instantiates
   a torchvision architecture from `model.arch` + `model.num_classes`,
   then calls `load_state_dict(strict=True)`.
3. **Pickled `nn.Module`** — if `torch.load` returns an `nn.Module`,
   RAITAP returns it directly and emits a `DeprecationWarning`. This path
   is supported but discouraged: pickled modules embed fully-qualified
   class paths, so renames or torchvision version bumps break unpickling.

## Config examples

```yaml
# A — full pickled nn.Module (deprecated):
model:
  source: "model.pth"

# B — state_dict + arch (recommended):
model:
  source: "weights.pth"
  arch: "resnet18"
  num_classes: 2
  pretrained: false      # default; set true to start from ImageNet weights

# C — TorchScript archive:
model:
  source: "scripted.pt"

# D — torchvision built-in (demo / quick test):
model:
  source: "resnet50"
```

## Migrating from pickled to state_dict

```python
m = torch.load("model.pth", weights_only=False)
torch.save(m.state_dict(), "weights.pth")
```

Then set `model.arch` and `model.num_classes` in your config to match the
architecture that produced the state-dict.

## ONNX

`.onnx` files are loaded via `OnnxBackend.from_path` — see
`src/raitap/models/backend.py` for the runtime selection logic. ONNX
backends are restricted to a subset of explainers (see the transparency
module README).
