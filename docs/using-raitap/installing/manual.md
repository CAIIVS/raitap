---
title: "Using manual deps management"
description: "Manage RAITAP's dependencies yourself with --custom-deps: pick a hardware backend extra (torch-cuda / torch-intel / torch-cpu) plus assessment extras, and re-declare the per-hardware index routing."
myst:
  html_meta:
    "description": "Manage RAITAP's dependencies yourself with --custom-deps: pick a hardware backend extra (torch-cuda / torch-intel / torch-cpu) plus assessment extras, and re-declare the per-hardware index routing."
---

# Using manual deps management

This page explains how to turn off automatic deps management and handle dependencies yourself.

## 1. Install the hardware backend

Install only the backend matching your setup.

```{install-tabs}
:uv:
uv add "raitap[torch-cuda]"

:pip:
pip install "raitap[torch-cuda]"
```

|       | CPU         | CUDA         | Intel GPU     |
| ----- | ----------- | ------------ | ------------- |
| Torch | `torch-cpu` | `torch-cuda` | `torch-intel` |
| ONNX  | `onnx-cpu`  | `onnx-cuda`  | `onnx-intel`  |

:::{note}
- CUDA = NVIDIA GPUs.
- `torch-intel` uses the Intel XPU API; `onnx-intel` uses the OpenVINO ONNX Runtime.
- Apple MPS support is coming soon.
:::

CUDA / Intel wheels do not live on PyPI. Re-declare the index routing per hardware:

:::{include} _gotchas.md
:::

(execution-dependencies)=

## 2. Pick assessment extras

(assessment-extras)=

You can either:

- Install a whole module (every library it offers):

  ```{install-tabs}
  :uv:
  uv add "raitap[transparency]"

  :pip:
  pip install "raitap[transparency]"
  ```

- Install a single library / report format. The extra is the library name (`captum`,
`mlflow`) or the format (`html`, `pdf`). You can combine it all in one line:

  ```{install-tabs}
  :uv:
  uv add "raitap[onnx-cpu,transparency,metrics]"

  :pip:
  pip install "raitap[onnx-cpu,transparency,metrics]"
  ```

## 3. Run the config with automatic deps management off

Pass <a href="../flags.html#flag-custom-deps"><code>--custom-deps</code></a>;
install everything the config needs first.

```{install-tabs}
:uv:
uv run raitap --config-dir my-configs --config-name assessment --custom-deps

:pip:
raitap --config-dir my-configs --config-name assessment --custom-deps
```
