# Configuration guide

This guide shows you how to configure RAITAP to assess your AI models using [Hydra](https://hydra.cc/) for flexible, reproducible configuration management.

## Quick start

RAITAP uses composable YAML configs that you can override via CLI:

```bash
# Use default settings
uv run raitap

# Use default settings but override all transparency options with the values in `configs/transparency/shap.yaml`
uv run raitap transparency=shap

# Use default settings but override specific options
uv run raitap transparency.algorithm=GradientExplainer

# Override data modality for tabular data
uv run raitap data.modality=tabular
```

## Full config structure

A RAITAP configuration has four main sections:

```yaml
# Optional: Use built-in models for testing (or set to null for your own, see below)
model:
  name: resnet50
  pretrained: true

# Your dataset configuration
data:
  name: isic2018
  description: "Skin lesion classification"
  directory: /path/to/data
  modality: image  # "image" or "tabular"

# What transparency framework & algorithm to use
transparency:
  framework: shap                    
  algorithm: GradientExplainer      
  output_dir: outputs/transparency

# Your experiment's unique name for tracking
experiment_name: my_assessment
```

### Config field reference

**Data config:**
- `name`: Dataset identifier (e.g., "isic2018", "my_dataset")
- `description`: Human-readable description
- `directory`: Path to dataset files (optional, for custom data loaders)
- `modality`: Data type - `"image"` for image data, `"tabular"` for feature vectors
  - Determines which visualiser is used (heatmap vs. bar chart)

**Transparency config:**

- `framework`: XAI library to use (`"captum"` or `"shap"`)
- `algorithm`: Method name (e.g., `"IntegratedGradients"`, `"GradientExplainer"`)
- `output_dir`: Where to save results

## Using RAITAP with your own model

### 1. Create your config

Create `my_config.yaml`:

```yaml
# Simple approach: specify all values directly
model:
  name: null  # Using your own model, not a built-in one

data:
  name: my_dataset
  description: "Production model assessment"
  directory: "path/to/my/dataset"
  modality: image  # "image" or "tabular"

transparency:
  framework: shap
  algorithm: GradientExplainer
  output_dir: "outputs/my_assessment"

experiment_name: production_audit_2026_Q1
```

<!-- markdownlint-disable-next-line MD033 -->
<details>
<!-- markdownlint-disable-next-line MD033 -->
<summary>Advanced config, using <code>defaults</code></summary>

The `defaults` section is loads and merges other config files.

```yaml
# Instead of writing this manually...
transparency:
  framework: shap
  algorithm: GradientExplainer

# ...you can tell Hydra to load configs/transparency/shap.yaml
defaults:
  - transparency: shap
```

RAITAP ships with built-in config files (in `configs/model/`, `configs/data/`, `configs/transparency/`) that you can reference this way. This is useful for composing configs. The simple approach above (specifying all values directly) is clearer for most use cases.

</details>

### 2. Write the assessment script

```python
import hydra
from omegaconf import DictConfig
import torch
from raitap.transparency import create_explainer, method_from_config

# Note: the path and name must match the file from step 1
@hydra.main(version_base="1.3", config_path=".", config_name="my_config")
def assess_model(cfg: DictConfig):
    # 1. Load your model
    model = torch.load("my_model.pth")
    model.eval()
    
    # 2. Load your data
    test_images = load_images(cfg.data.directory)
    
    # 3. Create explainer from config
    method = method_from_config(cfg.transparency)
    explainer = create_explainer(method, modality=cfg.data.modality)
    
    # 4. Generate explanations
    attributions = explainer.explain(model, test_images)
    
    # 5. Save results
    explainer.save(attributions, cfg.transparency.output_dir)

if __name__ == "__main__":
    assess_model()
```

### 3. Run assessment

```bash
uv run assess_model.py

# Or override arbitrary settings
uv run assess_model.py transparency.framework=captum
```

The results will be available in the directory specified in the config file (`output_dir`), with the following structure:

```text
outputs/captum/
└── 2026-02-23/
    └── 14-30-45/
        ├── attributions.pt        # Raw attribution values (PyTorch tensor)
        ├── image.png              # One file per requested visualiser
        └── metadata.json          # Config snapshot
```

### Tabular data example

For tabular/feature-based data, set `modality: tabular`:

```yaml
# tabular_config.yaml
model:
  name: null  # Your custom model

data:
  name: credit_risk
  description: "Credit risk prediction"
  directory: "data/credit/"
  modality: tabular  # Use bar chart visualiser

transparency:
  framework: shap
  algorithm: KernelExplainer
  output_dir: "outputs/tabular_assessment"
```

The visualiser will automatically generate feature importance bar charts instead of heatmaps.

---

## Advanced explainer configuration

### Explainers with constructor arguments

Some explainability methods require additional constructor arguments. For example, `LayerGradCam` needs a specific layer from your model:

```python
import torchvision.models as models
from raitap.transparency import create_explainer
from raitap.transparency.methods import Captum

# Load model
model = models.resnet50(pretrained=True)
model.eval()

# Create explainer with layer specification
explainer = create_explainer(
    Captum.LayerGradCam,
    modality="image",
    layer=model.layer4  # Constructor argument
)

# Use it normally
attributions = explainer.explain(model, images, target=0)
```

**How it works:** The factory passes `**kwargs` to the explainer constructor, which then forwards them to the underlying framework's method.

**Common use cases:**

- `LayerGradCam`: Requires `layer` parameter
- `GradientShap`: Accepts `multiply_by_inputs` (bool)
- SHAP explainers: May need `background_data` (passed to `.explain()` method instead)

```python
# Example: SHAP with background data
method = method_from_config(cfg.transparency)
explainer = create_explainer(method)

# Background data passed at explain time
background = test_images[:100]  # Subset for baseline
attributions = explainer.explain(
    model, 
    test_images,
    background_data=background
)
```

---

## Ensuring versioned, reproducible experiments

By creating different config files for each experiement, you allow for easy reproducibility and version control.

### `configs/baseline.yaml`

```yaml
# ... rest of the config, with specific settings ...

experiment_name: exp-023/baseline
```

### `configs/shap_comparison.yaml`

```yaml
# ... rest of the config, with different settings ...

experiment_name: exp-023/shap_comparison
```

Run specific experiments:

```bash
uv run raitap --config-name exp-023/baseline
uv run raitap --config-name exp-023/shap_comparison
```

## Batch testing (multirun)

You can also run multiple configs all at once.

```bash
uv run raitap --multirun \
    transparency=shap,captum \
    experiment_name=comparison_shap,comparison_captum
```

## Debugging (dry run)

You can view the current, final config without actually running the assessment:

```bash
uv run raitap --cfg job
```
