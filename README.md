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
    uv sync
    ```

### Basic Usage

#### In the CLI

```bash
# Run with default settings
uv run raitap

# Assess ResNet50 with SHAP explanations
uv run raitap model=resnet50 transparency=shap

# Try different transparency methods
uv run raitap transparency=captum transparency.algorithm=IntegratedGradients
```

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

### MLflow Tracking

To run an assessment with MLflow tracking enabled:

1. Start a local MLflow server:

   ```bash
   uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
   ```

2. In a second terminal, run RAITAP with MLflow tracking enabled:

   ```bash
   uv run raitap tracking=mlflow
   ```

   To override the tracking server explicitly:

   ```bash
   uv run raitap tracking=mlflow tracking.tracking_uri=http://127.0.0.1:5000
   ```

3. Open the MLflow dashboard:

   ```text
   http://127.0.0.1:5000
   ```

The run artifacts and metadata will be visible in the MLflow UI under the configured experiment.

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
