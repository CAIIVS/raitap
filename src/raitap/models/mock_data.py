"""Mock data generator for testing transparency methods."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def generate_mock_image(
    width: int = 224, height: int = 224, seed: int | None = None
) -> Image.Image:
    """
    Generate a random RGB image for testing.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        seed: Random seed for reproducibility

    Returns:
        PIL Image with random pixel values
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random RGB values (0-255)
    image_array = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

    return Image.fromarray(image_array, mode="RGB")


def generate_mock_batch(
    batch_size: int = 4, width: int = 224, height: int = 224, seed: int | None = None
) -> torch.Tensor:
    """
    Generate a batch of random images as a PyTorch tensor.

    Args:
        batch_size: Number of images in batch
        width: Image width in pixels
        height: Image height in pixels
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (batch_size, 3, height, width) with normalized values
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random tensor with values in [0, 1]
    # Shape: (batch_size, channels, height, width)
    batch = torch.rand(batch_size, 3, height, width)

    return batch


def generate_gradient_image(width: int = 224, height: int = 224) -> Image.Image:
    """
    Generate a gradient image for visual testing (easier to interpret attributions).

    Creates an image with horizontal gradient (dark to bright).

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        PIL Image with gradient pattern
    """
    # Create horizontal gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.tile(gradient, (height, 1))

    # Convert grayscale to RGB
    image_array = np.stack([gradient, gradient, gradient], axis=-1)

    return Image.fromarray(image_array, mode="RGB")
