# Installation

RAITAP infers the right `uv` extras from the Hydra config it is about to
run. In normal use you do not pick extras at all — `raitap` does it for
you and re-launches into a venv that has exactly what your config needs.

## 1. Install RAITAP

```{install-tabs}
:uv:
uv add raitap

:pip:
pip install raitap
```

:::{note}
RAITAP supports Python 3.11+ (3.11–3.13 tested). Python 3.14 is not yet
supported (Hydra 1.3.2 limitation). Some adapters (e.g. Marabou formal
verification) pin Python further; `raitap` picks a compatible interpreter
automatically when those adapters appear in the config.
:::

## 2. Run RAITAP

```bash
uv run raitap --config-dir my-configs --config-name assessment
```

That is the whole flow. `raitap` reads the config, infers the necessary
extras (backend, explainers, attackers, reporting, tracking, metrics),
probes the host hardware, picks the right Python, and re-launches itself
through `uv run` with the correct `--extra` flags. The first run prints a
panel like this before the install starts:

```
┌─ RAITAP · deps · sync + run ────────────────────────────────────────────┐
│                                                                         │
│  hardware  xpu (probed)                                                 │
│    python  host default                                                 │
│    extras  captum, jinja, metrics, torch-intel, torchattacks            │
│   command  uv sync --extra captum --extra jinja --extra metrics ...     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Preview without installing

Pass `--dry-run` to see the inferred command without syncing or running:

```bash
uv run raitap --dry-run --config-dir my-configs --config-name assessment
```

### Bypass inference

If you prefer to manage extras yourself (e.g. unusual driver setup, a
private fork that adds extras outside the RAITAP table), pass
`--custom-deps`:

```bash
uv run --extra torch-cuda --extra captum raitap \
    --custom-deps --config-dir my-configs --config-name assessment
```

`--custom-deps` skips the inference step entirely. Read the rest of this
page if you go that route — you are on the hook for choosing the right
backend and adapter extras.

(execution-dependencies)=

## Manual extras (advanced)

Use this section only when running with `--custom-deps`. The default flow
covers these decisions for you.

### Execution dependencies

RAITAP supports both PyTorch and ONNX models, and both CPU and GPU
execution. To avoid conflicts, only install the dependencies that match
your setup.

|       | CPU         | CUDA         | Intel GPU     |
| ----- | ----------- | ------------ | ------------- |
| Torch | `torch-cpu` | `torch-cuda` | `torch-intel` |
| ONNX  | `onnx-cpu`  | `onnx-cuda`  | `onnx-intel`  |

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

### Assessment dependencies

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
