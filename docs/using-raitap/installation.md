# Manual dependencies installation

This page explains how to manage config dependencies manually, without the automatic mode. It is recommended to use `uv`, but `pip` will also work.

## Enabling manual mode

Should you want to bypass the automatic deps detection and manage them yourself, you can pass the `--custom-deps` flag. RAITAP will not infer any dependencies from your config, and will not install anything. You are repsonsible for installing the necessary deps before running the config.

```{install-tabs}
:uv:
uv run raitap --config-dir my-configs --config-name assessment --custom-deps

:pip:
raitap --config-dir my-configs --config-name assessment --custom-deps
```

## Deciding which dependencies are needed for your config

(execution-dependencies)=

### Execution dependencies

RAITAP supports both PyTorch and ONNX models, and both CPU and GPU
execution. To avoid conflicts, only install the dependencies that match
your setup.

|       | CPU         | CUDA         | Intel GPU     |
| ----- | ----------- | ------------ | ------------- |
| Torch | `torch-cpu` | `torch-cuda` | `torch-intel` |
| ONNX  | `onnx-cpu`  | `onnx-cuda`  | `onnx-intel`  |

Combine with assessment extras as needed; see {ref}`assessment-extras` below.

:::{note}

- CUDA corresponds to NVIDIA GPUs.
- `torch-intel` uses the Intel XPU API directly.
- `onnx-intel` uses the OpenVINO ONNX Runtime.
- Apple MPS support is coming soon.
:::

:::{dropdown} Older NVIDIA GPUs and CUDA wheel selection
RAITAP does not force a single CUDA wheel family for packaged installs.
The right PyTorch CUDA wheels depend on your GPU generation and on which
CUDA wheel families a given PyTorch release supports.

This matters mainly for **older NVIDIA GPUs**, especially **Volta /
V100** systems, where a resolver may pick a newer CUDA wheel family that
is not a good match for the hardware even though the install itself
succeeds.

When in doubt:

- identify your GPU generation using NVIDIA's compatibility pages for
  [current GPUs](https://developer.nvidia.com/cuda/gpus) and
  [legacy GPUs](https://developer.nvidia.com/cuda/gpus/legacy)
- check the [PyTorch releases page](https://github.com/pytorch/pytorch/releases)
  for a release and CUDA wheel family that supports your hardware
- install the PyTorch CUDA packages from the matching PyTorch index, then
  install RAITAP and the remaining extras from PyPI

For example, on a **Volta / V100** system that needs the **`cu126`**
PyTorch wheels:

```{install-tabs}
:uv:
uv pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch torchvision
uv pip install "raitap[torch-cuda,transparency]"

:pip:
pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch torchvision
pip install "raitap[torch-cuda,transparency]"
```

This keeps PyTorch on the CUDA wheel family you selected while letting
RAITAP and the remaining dependencies resolve normally from PyPI.
:::

(assessment-extras)=

### Assessment dependencies

#### Module dependencies

Each module has its corresponding dependency group. it will install ALL libraries offered by the module.

```{install-tabs}
:uv:
uv add "raitap[transparency]"

:pip:
pip install "raitap[transparency]"
```

#### Library dependencies

You can also install specific libraries. The group's name is either the name of the library (e.g. `captum`, `mlflow`) or the report file fromat (`html`, `pdf`).

#### Combining in a one-liner

Combine as needed:

```{install-tabs}
:uv:
uv add "raitap[onnx-cpu,transparency,metrics]"

:pip:
pip install "raitap[onnx-cpu,transparency,metrics]"
```
