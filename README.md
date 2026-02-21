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

```bash
# Clone the repository
git clone https://github.zhaw.ch/RAI/Tech-Assessment-Platform.git
cd Tech-Assessment-Platform

# Install dependencies
uv sync
```

### Basic Usage

**As a CLI tool:**
```bash
# Run with default settings
uv run python -m raitap.run

# Assess ResNet50 with SHAP explanations
uv run python -m raitap.run model=resnet50 transparency=shap

# Try different transparency methods
uv run python -m raitap.run transparency=captum transparency.algorithm=saliency
```

**As a library (assess your own models):**
```python
import hydra
from raitap.transparency import create_explainer

@hydra.main(config_path="configs", config_name="config")
def assess_my_model(cfg):
    # Your custom model
    model = load_my_model()
    
    # Use raitap's explainability infrastructure
    explainer = create_explainer(cfg.transparency.framework, 
                                  cfg.transparency.algorithm)
    attributions = explainer.explain(model, my_data)
    explainer.save(attributions, cfg.transparency.output_dir)
```

### Documentation

- **[Configuration Guide](docs/configuration.md)** - Detailed config reference and usage patterns
- **[Development Guide](Development.md)** - MVP roadmap and technical details
- **[Contributing](CONTRIBUTING.md)** - How to contribute to the project

## Design Principles

### Easily Executable
- Simple CLI interface for quick experiments
- Sensible defaults - run with zero configuration
- Works as both CLI tool and Python library

### Easily Maintainable
- **Hydra**: Structured configuration management
- **uv**: Fast, reliable dependency management
- **Type-safe schemas**: Catch errors early with Python dataclasses
- **MLFlow**: Experiment tracking and model versioning (coming soon)

### Easily Extendable
- **Model-agnostic**: Works with any PyTorch `nn.Module`
- **Framework plugins**: Add new XAI methods via common interface
- **ONNX support**: Framework-independent model format (planned)
- **Modular architecture**: Swap components without breaking core functionality

### Key Technologies

- **[Hydra](https://hydra.cc/)** - Configuration framework with CLI integration
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager
- **[MLFlow](https://mlflow.org/)** - Experiment tracking and ML lifecycle management
- **[SHAP](https://shap.readthedocs.io/)** - Model explanations via Shapley values
- **[Captum](https://captum.ai/)** - PyTorch interpretability library

## Key Features

### Model-Agnostic Platform

RAITAP works with **any PyTorch model**. Whether you're using:
- Pretrained models (ResNet, ViT, Faster R-CNN)
- Your own custom architectures
- Fine-tuned models from production

The platform provides consistent explainability infrastructure for all.

### Flexible Configuration

Uses [Hydra](https://hydra.cc/) for configuration:
- **CLI Overrides**: Quick experiments without editing files
- **Config Composition**: Mix and match settings
- **Type-Safe**: Validated against Python schemas
- **Reproducible**: Every run logs its configuration

See the **[Configuration Guide](docs/configuration.md)** for details.

### Integrated Transparency Methods

**SHAP (SHapley Additive exPlanations)**
- GradientExplainer
- DeepExplainer (coming soon)
- KernelExplainer (coming soon)

**Captum (PyTorch)**
- Integrated Gradients
- Saliency (coming soon)
- GradCAM (coming soon)

More frameworks coming: OmniXAI, Alibi, custom methods.
