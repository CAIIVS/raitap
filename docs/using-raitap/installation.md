# Installation

This page explains how to install RAITAP and its dependencies.

<!-- :::{note}

If you are running RAITAP outside a Python project (no `pyproject.toml` in the directory), replace `uv sync` with `uv pip install` in the following.
::: -->

## 1. Install RAITAP

```{install-tabs}
:uv:
uv pip install raitap

:pip:
pip install raitap
```

## 2. Install dependencies

### Execution dependencies

RAITAP supports both PyTorch and ONNX models, and both CPU and GPU execution. Hence, you need to install the dependencies for your specific setup.

Here is an example command for PyTorch on CPU with ONNX support:

```{install-tabs}
:uv:
uv sync --extra onnx-cpu # deps for ONNX on CPU

:pip:
pip install "raitap[onnx-cpu]" # deps for ONNX on CPU
```

Here is the full list of dependency groups:

| Model format | CPU         | CUDA         | Intel         |
| ------------ | ----------- | ------------ | ------------- |
| Torch        | `torch-cpu` | `torch-cuda` | `torch-intel` |
| ONNX         | `onnx-cpu`  | `onnx-cuda`  | `onnx-intel`  |

:::{note}

- CUDA corresponds to NVIDIA GPUs.
- `torch-intel` uses the Intel XPU API directly.
- `onnx-intel` uses the OpenVINO ONNX Runtime.
- Apple MPS support is coming soon.
:::

### Assessment dependencies

You can then install the dependencies for the assessment modules you want to use.

For instance, if you want to assess the transparency, run:

```{install-tabs}
:uv:
uv sync --extra transparency

:pip:
pip install "raitap[transparency]"
```

If you plan to use a single underlying framework (here Captum), you can run the following instead:

```{install-tabs}
:uv:
uv sync --extra captum

:pip:
pip install "raitap[captum]"
```

### Single one-shot command

Of course, optional groups can be combined. For instance:

```{install-tabs}
:uv:
uv sync --extra onnx-cpu --extra transparency

:pip:
pip install "raitap[onnx-cpu,transparency]"
```
