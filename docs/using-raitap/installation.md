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

For projects using **`uv add` / a managed `pyproject.toml`** (instead of `uv pip install`), the equivalent declarative pattern requires both an index source mapping **and** promoting `torch` / `torchvision` to direct dependencies — uv only honors source overrides for direct deps, not transitive ones pulled in by `raitap[torch-cuda]`:

```toml
[project]
dependencies = [
  "raitap[torch-cuda, ...]>=0.4.2",
  # Direct-dep promotion is required for the source mapping below to fire.
  "torch>=2.10.0",
  "torchvision>=0.20.0",
]

[tool.uv.sources]
torch       = [{ index = "pytorch-cuda" }]
torchvision = [{ index = "pytorch-cuda" }]

[[tool.uv.index]]
name     = "pytorch-cuda"
url      = "https://download.pytorch.org/whl/cu126"
explicit = true
```

For most modern NVIDIA GPUs the routing is unnecessary — PyPI's default `torch` wheel ships with CUDA support and resolves transparently. Use this pattern only when you need a specific CUDA wheel family (cu118, cu121, cu126).

This is an example for older cards that need that wheel family. It is **not** the required default for every CUDA-capable install.
:::

:::{dropdown} Intel GPU (XPU) wheel selection
PyTorch's `+xpu` wheels (`torch`, `torchvision`) and `triton-xpu` are not on PyPI — they live on the Intel index at `download.pytorch.org/whl/xpu`. PyPI only hosts a yanked `triton-xpu==0.0.2`.

uv `[tool.uv.sources]` declared inside RAITAP do **not** propagate to consumers, so a project depending on RAITAP must redeclare the routing **and** promote `triton-xpu` to a direct dependency. uv only honors source overrides for direct deps; transitive `triton-xpu` (pulled in by `raitap[torch-intel] → torch+xpu`) is queried against PyPI alone and resolution fails.

In your project's `pyproject.toml`:

```toml
[project]
dependencies = [
  "raitap[torch-intel, ...]>=0.4.2; sys_platform != 'linux' and sys_platform != 'darwin'",
  # Promote triton-xpu to a direct dep so the source mapping below
  # routes it to the pytorch-intel index. Required — uv does not apply
  # source overrides to transitive deps.
  "triton-xpu>=3.0.0; sys_platform != 'linux' and sys_platform != 'darwin'",
]

[tool.uv.sources]
torch       = [{ index = "pytorch-intel" }]
torchvision = [{ index = "pytorch-intel" }]
triton-xpu  = [{ index = "pytorch-intel" }]

[[tool.uv.index]]
name     = "pytorch-intel"
url      = "https://download.pytorch.org/whl/xpu"
explicit = true
```

Then sync:

```{install-tabs}
:uv:
uv sync

:pip:
pip install --extra-index-url https://download.pytorch.org/whl/xpu \
  torch torchvision triton-xpu
pip install "raitap[torch-intel,transparency]"
```

Without the source mapping AND the direct-dep promotion, resolution fails with:

> Because only `triton-xpu<3.0.0` is available and `raitap[torch-intel]` depends on `triton-xpu>=3.0.0`, we can conclude that `raitap[torch-intel]` cannot be used.

The misleading hint about prereleases (`3.3.0b1`) is a symptom: uv only sees PyPI's yanked release line because the source override never fires for the transitive dep.
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
