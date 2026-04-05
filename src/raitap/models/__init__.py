"""
Models module, handles:

- loading models (pretrained or custom)
- selecting a backend for native PyTorch or ONNX execution.
"""

from .model import Model

__all__ = [
    "Model",
]
