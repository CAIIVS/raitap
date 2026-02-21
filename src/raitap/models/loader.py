"""
Convenience helpers for loading common pretrained models.

Note: The transparency platform works with ANY PyTorch nn.Module.
These helpers are optional - use them for quick prototyping/examples.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def load_pretrained_model(
    model_name: str, pretrained: bool = True, num_classes: int = 1000
) -> nn.Module:
    """
    Convenience function to load common pretrained models from torchvision.

    This is a helper for examples and quick prototyping. The raitap platform
    works with any PyTorch nn.Module - you can directly use your own models.

    Args:
        model_name: Name of the model (e.g., 'resnet50', 'vit_b_32')
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes (default 1000 for ImageNet)

    Returns:
        PyTorch model in evaluation mode

    Raises:
        ValueError: If model_name is not supported

    Example:
        >>> # For examples/tutorials
        >>> model = load_pretrained_model("resnet50")
        >>>
        >>> # For your own models
        >>> model = MyCustomModel()
        >>> # Both work the same in transparency methods!
    """
    model_name = model_name.lower()

    # Map model names to torchvision constructors
    model_registry = {
        "resnet50": models.resnet50,
        "vit_b_32": models.vit_b_32,
    }

    if model_name not in model_registry:
        raise ValueError(
            f"Model '{model_name}' not supported. Available models: {list(model_registry.keys())}"
        )

    # Load model with appropriate weights
    if pretrained:
        weights_param = "IMAGENET1K_V1"  # Default pretrained weights
        model = model_registry[model_name](weights=weights_param)
    else:
        model = model_registry[model_name](weights=None)

    # Set to evaluation mode (disable dropout, batch norm updates)
    model.eval()

    return model


def get_model_transform(model_name: str):
    """
    Get the preprocessing transform for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Transform function for preprocessing images
    """
    from torchvision import transforms

    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # All models expect 224x224 RGB input
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return transform
