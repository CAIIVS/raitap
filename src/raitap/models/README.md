# Models Module

This module loads machine-learning models into a uniform `ModelBackend`
that downstream RAITAP modules (transparency, metrics, reporting) consume
without caring about how the model was serialised.

It supports torchvision built-ins, ONNX archives, XGBoost tree models, and three flavours of
PyTorch checkpoint.

## Loader behaviour for `.pt` / `.pth`

Order of attempts when `model.source` points at a `.pt` or `.pth` file:

1. **TorchScript** (`torch.jit.load`) — if the file is a TorchScript
   archive, return the `ScriptModule` as-is. No extra config keys needed.
2. **state_dict** — if `torch.load` returns a `dict`, RAITAP instantiates
   a torchvision architecture from `model.arch` + `model.num_classes`,
   then calls `load_state_dict(strict=True)`.
3. **Pickled `nn.Module`** — if `torch.load` with `weights_only=True`
   rejects the checkpoint, RAITAP refuses it unless the caller supplied
   explicit consent at invocation time via the `allow_unsafe_pickle=True`
   kwarg on `raitap.run(...)` (Python API) or the `--allow-unsafe-pickle`
   CLI flag (surfaced through the `RAITAP_ALLOW_UNSAFE_PICKLE` env var the
   bootstrap exports across re-exec). With that explicit opt-in, RAITAP
   reloads with `weights_only=False`, returns the module directly, and
   emits a `DeprecationWarning`. This path executes arbitrary code
   embedded in the file and should only be used for fully trusted sources.

## Config examples

```yaml
# A — full pickled nn.Module (deprecated). Consent is supplied at invocation
# time, not in the config: pass `allow_unsafe_pickle=True` to
# `raitap.run(...)` or re-run the CLI with `--allow-unsafe-pickle`.
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

## XGBoost (`.ubj`)

`.ubj` files are loaded via `XGBoostBackend.from_path`. The backend provides
`TREE_MODEL` and `PREDICT_PROBA` capabilities, enabling `shap.TreeExplainer`
and model-agnostic SHAP explainers. Requires the `tree` extra:
`uv sync --extra tree`.
