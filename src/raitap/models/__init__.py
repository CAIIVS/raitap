"""
Models module, handles:

- loading models (pretrained or custom)
- converting from various formats to PyTorch nn.Module.
"""

from .model import Model

__all__ = [
    "Model",
]
