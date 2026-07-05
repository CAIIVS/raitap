---
title: "Configuration"
description: "This page describes how to configure the model that RAITAP will assess."
myst:
  html_meta:
    "description": "This page describes how to configure the model that RAITAP will assess."
---

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

:option: tokenizer
:allowed: string, null
:default: null
:description: HuggingFace hub id or local path for a tokenizer. Setting this
  selects the text input modality: `source` loads via
  `AutoModelForSequenceClassification` and `tokenizer` via `AutoTokenizer`.
  Requires the `text` extra (`uv sync --extra text`) and a `data.inputs`
  variant, see {doc}`/modules/data/own-vs-built-in`.

:yaml:
# Option A: single model file.
# `source` is the only key. The format is inferred from the extension
# (.pt/.pth, .onnx, .ubj). Per-format details and caveats (state-dict vs
# TorchScript vs pickled, the unsafe-pickle consent, the `--extra xgboost`
# requirement for .ubj) are on the "Using your own model" page (linked
# from the `source` option above).
model:
  source: "path/to/model.<ext>"

# Option B: state-dict file.
# A state-dict carries no architecture, so add `arch` + `num_classes` to
# rebuild the model before loading the weights.
model:
  source: "weights.pth"
  arch: "resnet18"
  num_classes: 2

# Option C: built-in torchvision model — `source` is the model name, not a
# path. Loaded with torchvision's `weights="DEFAULT"` (latest pretrained
# weights, not configurable). For demos / quick testing; load your own
# weights via a file path (Options A/B).
model:
  source: "resnet50"

# Option D: HuggingFace text model. `source` is a hub id or local path;
# `tokenizer` (usually the same id) selects the text input modality.
model:
  source: "distilbert-base-uncased-finetuned-sst-2-english"
  tokenizer: "distilbert-base-uncased-finetuned-sst-2-english"

:cli: model.source=resnet50
```
