# Installation

RAITAP uses `uv` for dependency management and exposes most optional functionality via
extras. Install exactly one Torch runtime profile, optionally add one ONNX profile, and
then layer feature extras on top.

## Prerequisites

- Python 3.13 or newer
- [`uv`](https://docs.astral.sh/uv/)

## Quick install

For a CPU-only setup, start with:

```bash
uv sync --extra torch-cpu
```

This installs the package plus the default runtime dependencies needed for the CLI and
the Python library.

## Torch runtime profiles

Choose exactly one Torch profile:

- CPU:

  ```bash
  uv sync --extra torch-cpu
  ```

- CUDA:

  ```bash
  uv sync --extra torch-cuda
  ```

- Intel XPU:

  ```bash
  uv sync --extra torch-xpu
  ```

## ONNX runtime profiles

Choose at most one ONNX profile. The current codebase still expects a Torch runtime
profile even for ONNX workflows, so combine one Torch extra with one ONNX extra.

- CPU:

  ```bash
  uv sync --extra torch-cpu --extra onnx-cpu
  ```

- GPU:

  ```bash
  uv sync --extra torch-cpu --extra onnx-gpu
  ```

- Intel / OpenVINO:

  ```bash
  uv sync --extra torch-cpu --extra onnx-openvino
  ```

## Feature extras

Add feature extras only when you need them:

- Captum:

  ```bash
  uv sync --extra torch-cpu --extra captum
  ```

- SHAP:

  ```bash
  uv sync --extra torch-cpu --extra shap
  ```

- Transparency bundle:

  ```bash
  uv sync --extra torch-cpu --extra transparency
  ```

- Metrics:

  ```bash
  uv sync --extra torch-cpu --extra metrics
  ```

- MLflow:

  ```bash
  uv sync --extra torch-cpu --extra mlflow
  ```

## Recommended local development setup

For documentation, tests, linting, and a CPU runtime, use:

```bash
uv sync --group dev --extra torch-cpu --extra transparency --extra metrics --extra mlflow
```

## What these extras control

- `torch-*`: selects the PyTorch runtime source and hardware profile
- `onnx-*`: installs the ONNX Runtime variant
- `captum` and `shap`: install explainability backends
- `transparency`: convenience bundle for `captum` plus `shap`
- `metrics`: enables TorchMetrics-based evaluation helpers
- `mlflow`: enables the MLflow tracking integration

For the next guided step, continue with [](first-run.md).
