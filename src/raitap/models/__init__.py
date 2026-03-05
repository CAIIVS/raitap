"""
Models module, handles:

- loading models (pretrained or custom)
- converting from various formats to PyTorch nn.Module.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .converters import CONVERTERS

__all__ = [
    "load_model",
]


def load_model(source: str) -> nn.Module:
    """
    Load a model from the source,
    which can be either a local file path or a torchvision model name.

    Args:
        source: Local model file path (e.g. ``model.pth``, ``model.onnx``)
                or a torchvision model name (e.g. ``"resnet50"``).

    Returns:
        PyTorch model in evaluation mode.

    Raises:
        FileNotFoundError: If *source* is a path that does not exist.
        ValueError: If *source* is an unsupported file format or an unknown
                    torchvision model name.
    """
    path = Path(source)
    suffix = path.suffix.lower()

    if path.exists() or suffix:
        return _load_from_path(path)

    name = str(source).lower()
    if hasattr(models, name):
        return _load_pretrained(name)

    raise ValueError(
        f"Model source {source!r} is neither an existing path nor a known "
        f"torchvision model.\n"
        f"Supported file formats: {list(CONVERTERS)}\n"
        f"To use your own model, set source to a valid model file path."
    )


def _load_from_path(path: Path) -> nn.Module:
    """
    Load a model from a file, converting foreign formats to ``.pth`` first.

    Args:
        path: Path to the model file.

    Returns:
        PyTorch model in evaluation mode.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension has no registered converter.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    converter = CONVERTERS.get(path.suffix.lower())
    if converter is None:
        raise ValueError(
            f"Unsupported model format {path.suffix!r}. Supported formats: {list(CONVERTERS)}"
        )

    pth_path = converter.convert(path)
    return _load_pth(pth_path)


def _load_pth(path: Path) -> nn.Module:
    """
    Load a PyTorch model from a ``.pth`` / ``.pt`` file saved with
    ``torch.save(model, path)``.

    Args:
        path: Path to the ``.pth`` / ``.pt`` file.

    Returns:
        PyTorch model in evaluation mode.

    Raises:
        ValueError: If the file contains a state-dict instead of a full model.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        raise ValueError(
            f"{path} appears to contain a state-dict, not a full model.\n"
            "Save the full model with torch.save(model, path) or load the "
            "state-dict manually and pass the model directly to the explainer."
        )

    if not isinstance(obj, nn.Module):
        raise ValueError(f"Expected an nn.Module in {path}, got {type(obj).__name__}.")

    obj.eval()
    return obj


def _load_pretrained(model_name: str) -> nn.Module:
    """
    Load a torchvision model with its default pre-trained weights.

    This is intended for demos and quick testing.  For production use, supply
    a model file path via :func:`load_model` instead.

    Args:
        model_name: Any ``torchvision.models`` attribute name
                    (e.g. ``"resnet50"``, ``"vit_b_32"``).

    Returns:
        PyTorch model in evaluation mode.

    Raises:
        ValueError: If *model_name* is not a valid torchvision model.
    """
    factory = getattr(models, model_name, None)
    if factory is None:
        raise ValueError(f"'{model_name}' is not a known torchvision model.")

    model = factory(weights="DEFAULT")  # type: ignore[operator]
    model.eval()
    return model
