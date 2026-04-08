<img src="assets/images/tech_assessment_platform_logo.png" width="400">

## Purpose

The certainty pipeline `raitap` is dedicated to the certification of responsible AI.
Four dimensions can be certified following MLOps practices.

The four dimensions are:

* Transparency
* Robustness
* Security
* Autonomy and control

For more information read

* [Towards the certification of AI-based systems](https://doi.org/10.1109/SDS60720.2024.00020)
* [MLOps as enable of trustworthy AI](https://doi.org/10.1109/SDS60720.2024.00013)

Additional references on XAI aspects:

* [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/).
* [One Explanation Does Not Fit All](https://doi.org/10.48550/arXiv.1909.03012)

## Quick Start

### Prerequisites

* Python 3.13 or higher
* [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository

    ```bash
    git clone https://github.zhaw.ch/RAI/Tech-Assessment-Platform.git
    cd Tech-Assessment-Platform
    ```

2. Install the dependencies

    ```bash
    uv sync --extra torch-cpu
    ```

#### Runtime profiles

Choose exactly one Torch runtime profile:

* CPU Torch:

    ```bash
    uv sync --extra torch-cpu
    ```

* CUDA Torch:

    ```bash
    uv sync --extra torch-cuda
    ```

* Intel XPU Torch:

    ```bash
    uv sync --extra torch-xpu
    ```

    This profile is currently locked for Python 3.13 while the published XPU dependency
    stack catches up for newer interpreter versions.

For ONNX workflows in the current codebase, also install a Torch runtime profile because ONNX
inputs and outputs still cross parts of the pipeline as `torch.Tensor`.

Choose at most one ONNX runtime profile:

* CPU:

    ```bash
    uv sync --extra torch-cpu --extra onnx-cpu
    ```

* GPU:

    ```bash
    uv sync --extra torch-cpu --extra onnx-cuda
    ```

* Intel / OpenVINO:

    ```bash
    uv sync --extra torch-cpu --extra onnx-intel
    ```

Add optional feature extras as needed:

```bash
uv sync --extra torch-cpu --extra captum --extra shap --extra metrics --extra mlflow
```

On Windows, `onnx-intel` also installs the `openvino` Python package. On Linux, the
OpenVINO ONNX Runtime wheel ships with the required OpenVINO libraries.

### Basic Usage

#### In the CLI

```bash
# Run with default settings
uv run raitap

# Force CPU execution
uv run raitap hardware=cpu

# Assess ResNet50 with SHAP explanations
uv run raitap model=resnet50 transparency=shap

# Try different transparency methods
uv run raitap transparency=captum transparency.algorithm=IntegratedGradients
```

### Runtime selection

RAITAP now uses a root-level `hardware` setting:

* Default: `hardware=gpu`
* Force CPU: `hardware=cpu`

When `hardware=gpu` is requested, RAITAP selects the best available supported accelerator and
falls back automatically when a higher-priority option is unavailable.

Priority order:

* PyTorch: `cuda` -> `xpu` -> `cpu`
* ONNX Runtime: `CUDAExecutionProvider` -> `OpenVINOExecutionProvider` -> `CPUExecutionProvider`

The runtime profiles above select the matching PyTorch wheel source for CPU, CUDA, or Intel XPU.
ONNX profiles remain separate and should not be combined with each other.

Apple GPU acceleration is temporarily disabled even when `hardware=gpu` is requested.
RAITAP currently falls back to CPU on Apple devices because the MPS/CoreML stack remains
immature for parts of the transparency pipeline, some explainer paths are unsupported,
and users have reported Apple GPU lockups / sustained 100% utilization after failures.

You can verify runtime availability with:

```python
import onnxruntime
import torch

print(torch.cuda.is_available())
print(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print(hasattr(torch, "xpu") and torch.xpu.is_available())
print(onnxruntime.get_available_providers())
```

If that snippet shows Apple MPS or `CoreMLExecutionProvider`, RAITAP still warns and
falls back to CPU today.

Accelerated ONNX execution helps inference and compatible non-autograd explainers. It does not add
PyTorch autograd to ONNX models.

#### In your Python code

```python
import torch
from raitap.transparency import explain

result = explain(config, my_model, my_input, target=0)
attributions = result["attributions"]  # torch.Tensor
figures = result["visualisations"]  # dict[str, Figure]
run_dir = result["run_dir"]  # pathlib.Path
```

For more details, see the [configuration guide](docs/consumers/configuration.md).

### Running With And Without MLflow

#### 1. Run without MLflow

Tracking is disabled by default.

```bash
uv run raitap
```

This writes local artifacts to the Hydra output directory printed in the console, for example:

```text
outputs/2026-03-13/13-21-24/
```

Use this mode when you only want local files and do not need an experiment dashboard.

#### 2. Start the bundled MLflow server, then run with MLflow

Start the local MLflow server in one terminal:

```bash
uv run raitap-mlflow-server
```

The launcher reads its defaults from `src/raitap/configs/tracking/mlflow_server.yaml` and starts a server on:

```text
http://127.0.0.1:5000
```

Then run RAITAP in a second terminal:

```bash
uv run raitap tracking=mlflow
```

Open the MLflow UI in your browser:

```text
http://127.0.0.1:5000
```

The local MLflow database and artifact store live under `./mlflow/`.

#### 3. Integrate with an existing MLflow setup

If you already have an MLflow tracking server, point RAITAP at that server explicitly:

```bash
uv run raitap tracking=mlflow tracking.tracking_uri=http://your-mlflow-host:5000
```

You can also override other tracking settings from the CLI, for example:

```bash
uv run raitap \
  tracking=mlflow \
  tracking.tracking_uri=http://your-mlflow-host:5000 \
  tracking.registry_uri=http://your-registry-host:5000 \
  tracking.log_model=true \
  experiment_name=my-assessment
```

Use this mode when MLflow is already managed outside this repository and you want RAITAP to publish runs into that existing environment.

#### What the main `raitap` command logs

When `tracking=mlflow` is enabled, the main CLI run logs:

- config snapshot
- dataset metadata
- transparency artifacts
- optional model artifact when `tracking.log_model=true`

The main `uv run raitap` flow currently does not compute or log evaluation metrics. If you need a complete MLflow smoke test including metrics, use:

```bash
uv run python -m raitap.tracking.smoke_test_mlflow
```

### Next steps

* If you want to contribute: [Contributing](CONTRIBUTING.md)

## Design Principles

### Easily Executable

* Simple CLI interface for quick experiments
* Hydra-based defaults and overrides for quick experimentation
* Works as both CLI tool and Python library

### Easily Maintainable

* **Hydra**: Structured configuration management
* **uv**: Fast, reliable dependency management
* **Type-safe schemas**: Catch errors early with Python dataclasses
* **MLflow**: Optional local experiment tracking and artifact logging

### Easily Extendable

* **Model-agnostic**: Works with any PyTorch `nn.Module`
* **Framework plugins**: Add new XAI methods via common interface
* **ONNX support**: Framework-independent model format (planned)
* **Modular architecture**: Swap components without breaking core functionality

### Key Technologies

* **[Hydra](https://hydra.cc/)** - Configuration framework with CLI integration
* **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager
* **[MLflow](https://mlflow.org/)** - Experiment tracking and ML lifecycle management
* **[SHAP](https://shap.readthedocs.io/)** - Model explanations via Shapley values
* **[Captum](https://captum.ai/)** - PyTorch interpretability library

## Key Features

### Model-Agnostic Platform

RAITAP works with **any PyTorch model**. Whether you're using:

* Pretrained models (ResNet, ViT, Faster R-CNN)
* Your own custom architectures
* Fine-tuned models from production

The platform provides consistent explainability infrastructure for all.

### Flexible Configuration

Uses [Hydra](https://hydra.cc/) for configuration:

* **CLI Overrides**: Quick experiments without editing files
* **Config Composition**: Mix and match settings
* **Type-Safe**: Validated against Python schemas
* **Reproducible**: Every run logs its configuration

See the **[Configuration Guide](docs/consumers/configuration.md)** for details.

### Integrated Transparency Methods

#### SHAP (SHapley Additive exPlanations)

* GradientExplainer
* DeepExplainer
* KernelExplainer
* TreeExplainer

#### Captum (PyTorch)

* Integrated Gradients
* Saliency
* LayerGradCam (GradCAM)
* DeepLift
* GuidedBackprop

More frameworks coming: OmniXAI, Alibi, custom methods.

## Metrics (TorchMetrics)

The metrics package defines a small interface for performance evaluation:

* `MetricComputer` with `reset()`, `update(predictions, targets)`, and `compute()`
* `MetricResult` with
    * `metrics: dict[str, float]` for scalar values
    * `artifacts: dict[str, Any]` for non-scalar outputs (lists, arrays, etc.)

Implementations:

* `ClassificationMetrics` (accuracy, precision, recall, f1)
* `DetectionMetrics` (mAP via `torchmetrics.detection.MeanAveragePrecision`)

### Programmatic usage

```python
import torch

from raitap.metrics.classification_metrics import ClassificationMetrics

preds = torch.tensor([0, 2, 1, 2])
targets = torch.tensor([0, 1, 1, 2])

metric = ClassificationMetrics(task="multiclass", num_classes=3, average="macro")
metric.update(preds, targets)
result = metric.compute()
print(result.metrics)
metric.reset()
```

### Detection usage

```python
import torch

from raitap.metrics.detection_metrics import DetectionMetrics

preds = [
    {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([1]),
    }
]
targets = [
    {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1]),
    }
]

metric = DetectionMetrics()
metric.update(preds, targets)
result = metric.compute()
print(result.metrics)
```

### Hydra config usage

Metrics configs live in `src/raitap/configs/metrics/`. Example override:

```bash
uv run python -m raitap.run metrics=classification metrics.num_classes=1000
```

To instantiate in code:

```python
from hydra.utils import instantiate

metric = instantiate(cfg.metrics)
```

Note: `src/raitap/run.py` currently does not instantiate metrics. Add the
`instantiate(cfg.metrics)` call where you compute metrics in your pipeline.
