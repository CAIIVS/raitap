# Installation

This page explains how to install RAITAP itself, and the deps needed to run any specific RAITAP config. It is recommended to use `uv`, but `pip` will also work.

## 1. Install RAITAP

First, you need to install the RAITAP package itself.

```{install-tabs}
:uv:
uv add raitap

:pip:
pip install raitap
```

:::{note}
RAITAP supports Python 3.11–3.13. Python 3.14 is not yet
supported (Hydra 1.3.2 limitation). Some underlying libs require older versions (e.g. Marabou < 3.12). RAITAP will handle the interpreter choice for you.
:::

## 2. Run a RAITAP config

RAITAP gives access to many underlying libraries (Captum, SHAP, Torchattacks...). To avoid bloat, they are not installed by default with RAITAP. Hence, the required dependencies are determined by analysing your config when you run it.

### Automatic mode (default)

You do not need to do anything but follow the instructions displayed in your terminal.

In the example below, we assume a basic config named `assessment`. For more details, see {doc}`configuration/index`.

```{install-tabs}
:uv:
uv run raitap --config-dir my-configs --config-name assessment

:pip:
raitap --config-dir my-configs --config-name assessment
```

Then, depending on your setup, RAITAP will either install deps automatically and start the run, or ask for further action:

- If you are using `uv`, it will ask you to run the `uv add` command yourself, or add the `--allow-project-edit` flag. This is because `uv add` modifies your `pyproject.toml`.
- If you are using `pip` and are not in a virtual environment (`venv`), it will ask to add the `--exec-global` flag. This will modify your global Python setup and is not recommended.

### Flags

The following flags are supported:

| Flag                    | Effect                                                                                             |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `--dry-run`             | Print the inferred plan, do not install, do not run                                                |
| `--sync-only`           | Install the inferred deps, do not run the pipeline                                                 |
| `--allow-project-edit`  | Consent to RAITAP modifying your project's `pyproject.toml` when adding deps                       |
| `--exec-global`         | Consent to `pip install` affecting your global Python (only used when no venv is detected)         |
| `--custom-deps`         | Skip automatic deps inference; rely on extras you installed manually (see below)                   |

(execution-dependencies)=

### Manual mode

Should you want to bypass the automatic deps detection and manage them yourself, you can pass the `--custom-deps` flag.

The following section explain how to choose your dependencies.

#### Execution dependencies

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

#### Assessment dependencies

Pick the extras that match the modules you use:

| Module    | Extra(s)                                  |
| --------- | ----------------------------------------- |
| Captum    | `captum` (or umbrella `transparency`)     |
| SHAP      | `shap` (or umbrella `transparency`)       |
| Torchattacks / Foolbox / Marabou | `torchattacks` / `foolbox` / `marabou` (or umbrella `robustness`) |
| Metrics   | `metrics`                                 |
| HTML report | `jinja` (or umbrella `reporting`)       |
| PDF report  | `borb` (or umbrella `reporting`)        |
| MLflow    | `mlflow` (or umbrella `tracking`)         |
| Slurm launcher | `launcher`                           |

Combine as needed:

```{install-tabs}
:uv:
uv add "raitap[onnx-cpu,transparency,metrics]"

:pip:
pip install "raitap[onnx-cpu,transparency,metrics]"
```
