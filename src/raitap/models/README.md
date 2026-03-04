# Models Module

## Design Philosophy

**The raitap platform works with ANY PyTorch `nn.Module`.**

This module provides **optional convenience helpers** for:

- Quick prototyping with common pretrained models
- Running examples and tutorials
- Generating mock data for testing

## Usage

### Option 1: Use Your Own Model (Recommended for Real Work)

```python
from raitap.transparency import create_explainer

# Bring your own model
my_model = MyCustomCNNModel()
my_model.eval()

# Use it directly with transparency methods
explainer = create_explainer("shap", "gradient")
attributions = explainer.explain(my_model, my_data)
```

### Option 2: Use Convenience Loaders (For Examples/Demos)

```python
from raitap.models import load_pretrained_model

# Quick way to get started with common models
model = load_pretrained_model("resnet50")

explainer = create_explainer("captum", "integrated_gradients")
attributions = explainer.explain(model, my_data)
```

## Files

- `loader.py` - Load common torchvision models (ResNet50, ViT, etc.)
- `mock_data.py` - Generate random/gradient images for testing
- Custom models work everywhere loaders work!

## Config

```yaml
# For examples: configs/model/resnet50.yaml
model:
  name: resnet50
  pretrained: true

# For custom models: configs/model/custom.yaml  
model:
  name: null
```
