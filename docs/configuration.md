# Configuration Guide

This guide explains how to configure the RAITAP platform for assessing AI models.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration Architecture](#configuration-architecture)
- [Configuration Reference](#configuration-reference)
- [Usage Patterns](#usage-patterns)
- [Advanced Topics](#advanced-topics)

---

## Quick Start

### Running with Defaults

```bash
uv run python -m raitap.run
```

### Override Specific Settings

```bash
# Use ResNet50 with SHAP explanations
uv run python -m raitap.run model=resnet50 transparency=shap

# Change SHAP algorithm
uv run python -m raitap.run transparency=shap transparency.algorithm=DeepExplainer

# Set custom output directory
uv run python -m raitap.run transparency.output_dir=outputs/my_experiment
```

---

## Configuration Architecture

RAITAP uses [Hydra](https://hydra.cc/) for configuration management, enabling:

- **Composition**: Mix and match config components
- **Override**: Change settings via CLI without editing files
- **Type Safety**: Validated against Python dataclass schemas
- **Reproducibility**: Every run logs its full configuration

### File Structure

```txt
src/raitap/configs/
├── config.yaml              # Main config (sets defaults)
├── schema.py                # Type definitions (source of truth)
├── model/                   # Model configurations
│   ├── resnet50.yaml
│   ├── vit_b32.yaml
│   └── custom.yaml          # For user's own models
├── data/                    # Dataset configurations
│   ├── isic2018.yaml
│   ├── malaria.yaml
│   └── udacityselfdriving.yaml
└── transparency/            # Explainability methods
    ├── shap.yaml
    └── captum.yaml
```

### How Composition Works

When you run `python -m raitap.run model=resnet50 transparency=shap`:

1. Hydra loads `config.yaml` (base configuration)
2. Merges in `model/resnet50.yaml`
3. Merges in `transparency/shap.yaml`
4. Applies any CLI overrides
5. Validates against schemas in `schema.py`
6. Returns a single `AppConfig` object

---

## Configuration Reference

All configurations are defined as Python dataclasses in [`schema.py`](../src/raitap/configs/schema.py). This is the **source of truth** for available options.

### Model Configuration

**Purpose:** Optional helper for loading common pretrained models. **The platform works with ANY PyTorch model** - this config is only for convenience.

```yaml
model:
  name: resnet50        # Model name or null for custom models
  pretrained: true      # Load pretrained weights?
```

**Available Models:**

- `resnet50` - ResNet-50 (ImageNet pretrained)
- `vit_b_32` - Vision Transformer Base/32 (ImageNet pretrained)
- `null` - Use your own custom model

**Schema:**

```python
@dataclass
class ModelConfig:
    name: str | None = None
    pretrained: bool = True
```

**Example - Using Custom Model:**

```yaml
model:
  name: null  # Ignore convenience loaders, use your own
```

---

### Data Configuration

**Purpose:** Specify which dataset to use for assessment.

```yaml
data:
  name: isic2018                                    # Dataset identifier
  description: "ISIC 2018 skin lesion dataset"     # Human-readable description
  directory: /path/to/data                          # Optional: data location
```

**Available Datasets:**

- `isic2018` - Dermatology (skin lesion classification)
- `malaria` - Malaria parasite detection
- `udacityselfdriving` - Self-driving car object detection

**Schema:**

```python
@dataclass
class DataConfig:
    name: str = "isic2018"
    description: str | None = None
    directory: str | None = None
```

---

### Transparency Configuration

**Purpose:** Configure which explainability method to use for model assessment.

```yaml
transparency:
  framework: shap                          # Which XAI framework to use
  algorithm: GradientExplainer             # Specific method within framework
  output_dir: outputs/transparency         # Where to save results
```

#### Framework Options

**`shap`** - [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/)

- **Algorithms:**
  - `GradientExplainer` - Fast, gradient-based (recommended for neural networks)
  - `DeepExplainer` - Deep learning optimized
  - `KernelExplainer` - Model-agnostic (slower, works with any model)

**`captum`** - [PyTorch Captum](https://captum.ai/)

- **Algorithms:**
  - `integrated_gradients` - Integrated Gradients (default)
  - `saliency` - Gradient-based saliency maps
  - `gradcam` - Gradient-weighted Class Activation Mapping (CNN specific)
  - `deeplift` - DeepLIFT attribution

**Schema:**

```python
class TransparencyFramework(StrEnum):
    captum = "captum"
    shap = "shap"

@dataclass
class TransparencyConfig:
    framework: TransparencyFramework = TransparencyFramework.captum
    algorithm: str = "integrated_gradients"
    output_dir: str = "outputs/transparency"
```

#### Output Structure

Results are saved with timestamps for traceability:

```text
outputs/
└── 2026-02-21/
    └── 20-14-24/
        ├── attributions.npy       # Raw attribution values
        ├── visualization.png      # Human-readable visualization
        └── metadata.json          # Full config used for this run
```

---

### Experiment Configuration

**Purpose:** Name your experiment for tracking and organization.

```yaml
experiment_name: mvp  # Used in output paths and MLFlow tracking
```

**Schema:**

```python
@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transparency: TransparencyConfig = field(default_factory=TransparencyConfig)
    experiment_name: str = "mvp"
```

---

## Usage Patterns

### Pattern 1: Quick Experiments (CLI)

**Use Case:** Testing different explainability methods on pretrained models.

```bash
# Try different transparency methods
uv run python -m raitap.run transparency=shap
uv run python -m raitap.run transparency=captum

# Try different models
uv run python -m raitap.run model=resnet50 transparency=shap
uv run python -m raitap.run model=vit_b32 transparency=shap

# Change algorithm within a framework
uv run python -m raitap.run transparency=shap transparency.algorithm=KernelExplainer
```

---

### Pattern 2: Assess Your Own Model (Library)

**Use Case:** Production model assessment for auditing/certification.

**Your Project Structure:**

```txt
my_project/
├── configs/
│   └── config.yaml
├── assess_model.py
├── my_model.pth
└── data/
    └── test_images/
```

**Your `configs/config.yaml`:**

```yaml
# Model config (ignored when using custom model)
model:
  name: null

# Your dataset
data:
  name: custom_dataset
  description: "Production melanoma classifier v2.3"
  directory: "data/test_images"

# Transparency settings (use raitap infrastructure)
transparency:
  framework: shap
  algorithm: GradientExplainer
  output_dir: "outputs/audit_2026_Q1"

# Experiment tracking
experiment_name: melanoma_v2.3_certification
```

**Your `assess_model.py`:**

```python
import hydra
from omegaconf import DictConfig
import torch
from raitap.transparency import create_explainer
from my_models import MelanomaClassifier

@hydra.main(version_base="1.3", 
            config_path="configs",
            config_name="config")
def assess_my_model(cfg: DictConfig):
    # Load your custom model
    model = MelanomaClassifier(num_classes=7)
    model.load_state_dict(torch.load("my_model.pth"))
    model.eval()
    
    # Load your data
    images = load_test_images(cfg.data.directory)
    
    # Use raitap's explainability infrastructure
    explainer = create_explainer(
        cfg.transparency.framework,
        cfg.transparency.algorithm
    )
    
    # Generate explanations
    attributions = explainer.explain(model, images)
    
    # Save for audit
    explainer.save(attributions, cfg.transparency.output_dir)
    
    print(f"✅ Assessment complete: {cfg.transparency.output_dir}")

if __name__ == "__main__":
    assess_my_model()
```

**Run:**

```bash
cd my_project
python assess_model.py
```

---

### Pattern 3: Config Files for Reproducibility

**Use Case:** Running multiple experiments with version-controlled configurations.

**Project Structure:**

```txt
experiments/
├── configs/
│   ├── baseline.yaml
│   ├── shap_comparison.yaml
│   └── captum_comparison.yaml
└── run_experiments.py
```

**`configs/baseline.yaml`:**

```yaml
model:
  name: resnet50
  pretrained: true

data:
  name: isic2018

transparency:
  framework: captum
  algorithm: integrated_gradients
  output_dir: outputs/baseline

experiment_name: baseline_experiment
```

**`configs/shap_comparison.yaml`:**

```yaml
defaults:
  - baseline

transparency:
  framework: shap
  algorithm: GradientExplainer
  output_dir: outputs/shap_comparison

experiment_name: shap_comparison
```

**Run specific configs:**

```bash
uv run python -m raitap.run --config-name baseline
uv run python -m raitap.run --config-name shap_comparison
```

---

## Advanced Topics

### Nested Overrides

Override deeply nested values:

```bash
# Change multiple transparency settings
uv run python -m raitap.run \
    transparency.framework=shap \
    transparency.algorithm=DeepExplainer \
    transparency.output_dir=outputs/deep_shap
```

### Multirun (Sweeps)

Run multiple configurations automatically:

```bash
# Test all combinations
uv run python -m raitap.run --multirun \
    model=resnet50,vit_b32 \
    transparency=shap,captum
```

This creates 4 runs:

- resnet50 + shap
- resnet50 + captum
- vit_b32 + shap
- vit_b32 + captum

### Config Groups

Create your own config groups:

```yaml
# configs/transparency/my_custom_shap.yaml
framework: shap
algorithm: KernelExplainer
output_dir: outputs/custom_kernel_shap
```

Use it:

```bash
uv run python -m raitap.run transparency=my_custom_shap
```

### Printing Current Config

See what configuration will be used:

```bash
uv run python -m raitap.run --cfg job
```

---

## Schema Validation

All configs are validated against Python dataclasses. If you provide invalid values, you'll get clear error messages:

```bash
# Invalid framework
uv run python -m raitap.run transparency.framework=invalid
# Error: 'invalid' is not a valid TransparencyFramework

# Wrong type
uv run python -m raitap.run model.pretrained=yes
# Error: Expected bool, got str
```

---

## Configuration Tips

1. **Start Simple**: Use CLI overrides for exploration
2. **Create Config Files**: For reproducible experiments
3. **Version Control Configs**: Track experiment settings in Git
4. **Use Type Hints**: IDEs will autocomplete config options
5. **Check Schema**: [`schema.py`](../src/raitap/configs/schema.py) is the source of truth

---

## Getting Help

- **Schema Reference**: See [`src/raitap/configs/schema.py`](../src/raitap/configs/schema.py)
- **Example Configs**: Browse `src/raitap/configs/*/`
- **Hydra Docs**: [https://hydra.cc/](https://hydra.cc/)
- **Report Issues**: [GitHub Issues](https://github.zhaw.ch/RAI/Tech-Assessment-Platform/issues)
