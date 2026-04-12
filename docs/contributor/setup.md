# Development environment setup

## Prerequisites

- The repository cloned.
- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager

## Installing dependencies

RAITAP supports many machine and model configurations. To avoid conflicts, only install the dependencies that match your setup. Linting and testing of all configurations is done on the CI.

1. Choose your execution dependency group from the following table:

    |       | CPU         | CUDA         | Intel         |
    | ----- | ----------- | ------------ | ------------- |
    | Torch | `torch-cpu` | `torch-cuda` | `torch-intel` |
    | ONNX  | `onnx-cpu`  | `onnx-cuda`  | `onnx-intel`  |

    :::{note}

    - CUDA corresponds to NVIDIA GPUs.
    - `torch-intel` uses the Intel XPU API directly.
    - `onnx-intel` uses the OpenVINO ONNX Runtime.
    - Apple MPS support is coming soon.
    :::

2. Decide which optional dependencies you need. It can either be a whole module (e.g. `transparency`, `metrics`, `tracking`) or a specific framework/integration (e.g. `shap`, `captum`, `mlflow`).

3. Consolidate all the dependency groups into a single command and run it. Notice the `--group dev`   flag to install the contributor environment. Here an example:

    ```shell
    uv sync --group dev --extra onnx-cpu --extra transparency
    ```

    :::{warning}
    Do not run the `sync` commands separately. The latest run will override the previous ones.
    :::
