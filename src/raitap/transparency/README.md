# RAITAP Transparency Module

A flexible, maintainable explainability module for PyTorch models that wraps SHAP and Captum frameworks.

## Features

- ✅ **Type-safe API** - Registry pattern prevents invalid method combinations
- ✅ **IDE-friendly** - Full autocomplete and navigation support
- ✅ **Separation of concerns** - Attribution computation ≠ Visualization
- ✅ **Config integration** - Seamless Hydra workflow
- ✅ **Extensible** - Add new methods with one-line changes
- ✅ **Production-ready** - Proper error handling and edge cases

## Quick Start

### Basic Usage (Registry API)

```python
from raitap.transparency import create_explainer
from raitap.transparency.methods import Captum
from raitap.transparency.visualisers import ImageHeatmapvisualiser
import torch

# Load model and data
model = load_my_model()
images = torch.randn(8, 3, 224, 224)

# Create explainer (type-safe, IDE autocomplete)
explainer = create_explainer(Captum.IntegratedGradients, modality="image")

# Compute attributions
attributions = explainer.explain(model, images, target=0)

# Visualize
visualiser = ImageHeatmapvisualiser()
visualiser.save(attributions, "outputs/attributions.png", inputs=images)
```

### Config-Driven Workflow (Hydra Integration)

```yaml
# configs/transparency/captum.yaml
framework: captum
algorithm: IntegratedGradients
output_dir: outputs/transparency
```

```python
from raitap.transparency import create_explainer, method_from_config

# Translate config → registry
method = method_from_config(
    cfg.transparency.framework,
    cfg.transparency.algorithm
)

# Create and use explainer
explainer = create_explainer(method, modality="image")
attributions = explainer.explain(model, data, target=0)
```

## Architecture

### Design Pattern: Separation of Concerns

```
┌─────────────────────────────────────────────┐
│         User-Facing API                     │
│  create_explainer() factory                 │
└────────────┬────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼──────┐  ┌──────▼────────┐
│Explainers│  │ visualisers   │
│(Framework)  │  (Modality)   │
└──────────┘  └───────────────┘
```

### Components

1. **Method Registry** (`methods.py`)
   - Type-safe, zero-duplication method definitions
   - IDE autocomplete support
   - Curated, tested methods only

2. **Explainers** (`explainers/`)
   - `CaptumExplainer` - Single class for all Captum methods
   - `ShapExplainer` - Single class for all SHAP explainers
   - Dynamic method loading (no class explosion)

3. **visualisers** (`visualisers/`)
   - `ImageHeatmapvisualiser` - For image inputs
   - `TabularBarChartvisualiser` - For tabular data
   - Modality-specific, framework-agnostic

4. **Factory** (`factory.py`)
   - `create_explainer()` - Main factory function
   - `method_from_config()` - Config bridge for Hydra

## Supported Methods

### Captum
- `IntegratedGradients`
- `Saliency`
- `GradCAM`
- `DeepLift`
- `GuidedBackprop`

### SHAP
- `GradientExplainer`
- `DeepExplainer`
- `KernelExplainer`
- `TreeExplainer`

## Adding New Methods

### Step 1: Test Compatibility

```python
# tests/transparency/test_captum_explainer.py
def test_new_method_with_visualiser():
    explainer = CaptumExplainer("NewMethod")
    attributions = explainer.explain(model, images, target=0)
    
    visualiser = ImageHeatmapvisualiser()
    visualiser.save(attributions, "test_output.png", inputs=images)
    # Manual inspection: Does it look correct?
```

### Step 2: Add to Registry (ONE LINE)

```python
# transparency/methods.py
class Captum:
    __framework__ = "captum"
    
    IntegratedGradients = ExplainerMethod()
    Saliency = ExplainerMethod()
    NewMethod = ExplainerMethod()  # ← ONE LINE CHANGE
```

### Step 3: Use Immediately

```python
from raitap.transparency.methods import Captum
explainer = create_explainer(Captum.NewMethod, modality="image")
```

## Testing

Run the test suite:

```bash
pytest tests/transparency/ -v
```

Test coverage:
- ✅ Registry correctness
- ✅ Explainer implementations
- ✅ visualisers
- ✅ Error handling
- ✅ End-to-end workflows

## Dependencies

Required:
- `torch>=2.0.0`
- `matplotlib>=3.5.0`

Optional (install as needed):
- `captum>=0.7.0` - For Captum methods
- `shap>=0.46.0` - For SHAP explainers

## Examples

See `examples/` directory:
- `transparency_basic.py` - Basic registry API usage
- `transparency_config.py` - Config-driven workflow

## Error Handling

### Invalid Method (Caught at Runtime)

```python
from raitap.transparency.methods import Captum

explainer = create_explainer(Captum.InvalidMethod)
# AttributeError: 'Captum' has no attribute 'InvalidMethod'
# ✅ Clear error before execution
```

### Typo in Config

```python
method = method_from_config("captum", "IntegratedGradient")  # Missing 's'
# ValueError: Captum has no method 'IntegratedGradient' in RAITAP registry.
# Available methods: IntegratedGradients, Saliency, GradCAM, ...
# ✅ Helpful error with suggestions
```

## API Reference

See inline docstrings for detailed documentation:

```python
from raitap.transparency import create_explainer
help(create_explainer)
```

## License

Part of the RAITAP project.
