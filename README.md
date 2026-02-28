<!-- markdownlint-disable-next-line MD033 MD045 MD041 -->
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
from raitap.transparency import create_explainer
from raitap.transparency.methods import Captum, SHAP
from raitap.transparency.visualisers import ImageHeatmapvisualiser

explainer = create_explainer(Captum.IntegratedGradients, modality="image")
attributions = explainer.explain(my_model, my_input, target=0)

visualiser = ImageHeatmapvisualiser()
visualiser.save(attributions, "outputs/attributions.png", inputs=my_input)
```

See [examples/](examples/) for more.

#### With a [Hydra](https://hydra.cc/) config

For reproducible experiments and CLI workflows, use a [Hydra](https://hydra.cc/) configuration:

```python
import hydra
from raitap.transparency import create_explainer, method_from_config
from raitap.transparency.visualisers import ImageHeatmapvisualiser

@hydra.main(config_path="configs", config_name="config")
def assess_my_model(cfg):
    
    # Translate config to registry
    method = method_from_config(
        cfg.transparency.framework,
        cfg.transparency.algorithm
    )
    
    # Create explainer from config
    explainer = create_explainer(method, modality="image")
    attributions = explainer.explain(my_model, my_input, target=0)
    
    # Visualize and save
    visualiser = ImageHeatmapvisualiser()
    visualiser.save(attributions, f"{cfg.transparency.output_dir}/result.png")
```

For more details, see the [configuration guide](docs/consumers/configuration.md).

### Next steps

* If you want to contribute: [Contributing](CONTRIBUTING.md)

## Design Principles

### Easily Executable

* Simple CLI interface for quick experiments
* Sensible defaults - run with zero configuration
* Works as both CLI tool and Python library

### Easily Maintainable

* **Hydra**: Structured configuration management
* **uv**: Fast, reliable dependency management
* **Type-safe schemas**: Catch errors early with Python dataclasses
* **MLFlow**: Experiment tracking and model versioning (coming soon)

### Easily Extendable

* **Model-agnostic**: Works with any PyTorch `nn.Module`
* **Framework plugins**: Add new XAI methods via common interface
* **ONNX support**: Framework-independent model format (planned)
* **Modular architecture**: Swap components without breaking core functionality

### Key Technologies

* **[Hydra](https://hydra.cc/)** - Configuration framework with CLI integration
* **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager
* **[MLFlow](https://mlflow.org/)** - Experiment tracking and ML lifecycle management
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

See the **[Configuration Guide](docs/configuration.md)** for details.

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
