"""
Models module: convenience helpers for examples and test data.

The raitap platform works with any PyTorch nn.Module. These utilities are
optional helpers for quick prototyping and examples.
"""

from .loader import get_model_transform, load_pretrained_model
from .mock_data import generate_gradient_image, generate_mock_batch, generate_mock_image

__all__ = [
    "generate_gradient_image",
    "generate_mock_batch",
    "generate_mock_image",
    "get_model_transform",
    "load_pretrained_model",
]
