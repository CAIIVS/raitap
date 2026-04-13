# Installation

This page explains how to install RAITAP from PyPI and how to select the optional dependency groups you need.

Use `uv add` if you are working inside a managed Python project (`pyproject.toml` exists). Use `uv pip install` or `pip install` otherwise.

## 1. Install RAITAP

```{install-tabs}
:uv:
uv add raitap

:pip:
pip install raitap
```

## 2. Install optional dependencies

(execution-dependencies)=

### Execution dependencies

RAITAP supports both PyTorch and ONNX models, and both CPU and GPU execution. To avoid conflicts, only install the dependencies that match your setup.

First, choose the right group from this table:

|       | CPU         | CUDA         | Intel GPU     |
| ----- | ----------- | ------------ | ------------- |
| Torch | `torch-cpu` | `torch-cuda` | `torch-intel` |
| ONNX  | `onnx-cpu`  | `onnx-cuda`  | `onnx-intel`  |

Then, adapt the following example command and run it:

```{install-tabs}
:uv:
uv add "raitap[onnx-cpu]" # replace `onnx-cpu` with your group

:pip:
pip install "raitap[onnx-cpu]" # replace `onnx-cpu` with your group
```

:::{note}

- CUDA corresponds to NVIDIA GPUs.
- `torch-intel` uses the Intel XPU API directly.
- `onnx-intel` uses the OpenVINO ONNX Runtime.
- Apple MPS support is coming soon.
:::

### Assessment dependencies

You can then install the dependencies for the assessment modules you want to use.

For instance, if you want to assess the model's transparency, run:

```{install-tabs}
:uv:
uv add "raitap[transparency]"

:pip:
pip install "raitap[transparency]"
```

If you plan to use a single underlying framework (here Captum), you can run the following instead:

```{install-tabs}
:uv:
uv add "raitap[captum]"

:pip:
pip install "raitap[captum]"
```

:::{note}
**Alibi** requires dependency overrides before it can be installed. See {ref}`Alibi (transparency) <alibi-frameworks>` for the exact steps.
:::

### Combine multiple extras

Of course, such optional dependency groups can be combined. For instance:

```{install-tabs}
:uv:
uv add "raitap[onnx-cpu,transparency]"

:pip:
pip install "raitap[onnx-cpu,transparency]"
```
