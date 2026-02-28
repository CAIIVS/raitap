"""Utilities for loading and preprocessing image data."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def load_images_from_directory(directory: str | Path, size: int = 224) -> torch.Tensor:
    """
    Load all images from a directory and preprocess them for ImageNet models.

    Applies standard ImageNet preprocessing: resize → center-crop → normalize.

    Args:
        directory: Path to a directory containing image files.
        size: Target crop size (default 224 for most ImageNet models).

    Returns:
        Tensor of shape (N, 3, size, size) with normalised values.

    Raises:
        FileNotFoundError: If no images are found in the directory.
    """
    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    directory = Path(directory)
    image_files = sorted(
        f
        for f in directory.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {directory}")

    tensors = []
    for f in image_files:
        img = Image.open(f).convert("RGB")
        tensors.append(preprocess(img))

    return torch.stack(tensors)
