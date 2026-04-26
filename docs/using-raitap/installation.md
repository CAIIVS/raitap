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

:::{note}
RAITAP was currently tested with Python 3.13.x. Ensure your project matches this requirement, or expect possible issues.
:::

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

:::{dropdown} Older NVIDIA GPUs and CUDA wheel selection
RAITAP does not force a single CUDA wheel family for packaged installs. The right PyTorch CUDA wheels depend on your GPU generation and on which CUDA wheel families a given PyTorch release supports.

This matters mainly for **older NVIDIA GPUs**, especially **Volta / V100** systems, where a resolver may pick a newer CUDA wheel family that is not a good match for the hardware even though the install itself succeeds.

When in doubt:

- identify your GPU generation using NVIDIA's compatibility pages for [current GPUs](https://developer.nvidia.com/cuda/gpus) and [legacy GPUs](https://developer.nvidia.com/cuda/gpus/legacy)
- check the [PyTorch releases page](https://github.com/pytorch/pytorch/releases) for a release and CUDA wheel family that supports your hardware
- install the PyTorch CUDA packages from the matching PyTorch index, then install RAITAP and the remaining extras from PyPI

For example, if you are installing RAITAP on a **Volta / V100** system and want to use the **`cu126`** PyTorch wheels, the most robust approach is to install in **two steps**:

```{install-tabs}
:uv:
uv pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch torchvision
uv pip install "raitap[torch-cuda,transparency]"

:pip:
pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch torchvision
pip install "raitap[torch-cuda,transparency]"
```

This keeps the PyTorch packages on the CUDA wheel family you selected, while letting RAITAP and the remaining dependencies resolve normally from PyPI.

If you also need extras such as `metrics`, the same two-step pattern still applies:

```{install-tabs}
:uv:
uv pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch torchvision
uv pip install "raitap[launcher,transparency,metrics,reporting]"

:pip:
pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch torchvision
pip install "raitap[launcher,transparency,metrics,reporting]"
```

This came up in cluster testing with `metrics`, where `torchmetrics` should resolve from PyPI while `torch` / `torchvision` should come from the PyTorch CUDA index.

This is an example for older cards that need that wheel family. It is **not** the required default for every CUDA-capable install.
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

### Combine multiple extras

Of course, such optional dependency groups can be combined. For instance:

```{install-tabs}
:uv:
uv add "raitap[onnx-cpu,transparency]"

:pip:
pip install "raitap[onnx-cpu,transparency]"
```
