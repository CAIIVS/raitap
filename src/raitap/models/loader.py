"""
Convenience helpers for loading common pretrained models.

Note: The transparency platform works with ANY PyTorch nn.Module.
These helpers are optional - use them for quick prototyping/examples.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

# Built-in torchvision shortcuts for demos/quick testing.
# weights="DEFAULT" resolves to the best available checkpoint per architecture.
_MODEL_REGISTRY: dict[str, object] = {
    "resnet50": models.resnet50,
    "vit_b_32": models.vit_b_32,
}


def load_model(source: str | Path) -> nn.Module:
    """
    Load a model from *source*, using the same resolution logic as
    :func:`~raitap.data.loader.resolve_data_source`:

    1. ``Path(source).exists()`` or extension is ``.pth``/``.pt`` ->
       :func:`load_model_from_path`.
    2. Known name in the built-in registry -> :func:`load_pretrained_model`.
    3. Otherwise -> :exc:`ValueError`.

    Args:
        source: Local ``.pth`` path or a built-in name (e.g. ``"resnet50"``).

    Returns:
        PyTorch model in evaluation mode.
    """
    path = Path(source)
    if path.exists() or path.suffix.lower() in {".pth", ".pt"}:
        return load_model_from_path(path)

    name = str(source).lower()
    if name in _MODEL_REGISTRY:
        return load_pretrained_model(name)

    known = list(_MODEL_REGISTRY.keys())
    raise ValueError(
        f"Model source {source!r} is neither an existing path nor a known built-in name.\n"
        f"Built-in names: {known}\n"
        f"To use your own model, set source to a valid .pth file path."
    )


def load_pretrained_model(model_name: str) -> nn.Module:
    """
    Load a built-in torchvision model with its default pre-trained weights.

    This is a helper for demos and quick testing. For production use, supply
    a ``.pth`` path via :func:`load_model` instead.

    Args:
        model_name: Name of the model (e.g. ``'resnet50'``, ``'vit_b_32'``).

    Returns:
        PyTorch model in evaluation mode.

    Raises:
        ValueError: If *model_name* is not in the built-in registry.
    """
    model_name = model_name.lower()

    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not in built-in registry: {list(_MODEL_REGISTRY.keys())}"
        )

    model = _MODEL_REGISTRY[model_name](weights="DEFAULT")  # type: ignore[operator]
    model.eval()
    return model


def load_model_from_path(path: str | Path) -> nn.Module:
    """
    Load a PyTorch model from a ``.pth`` file saved with ``torch.save``.

    Handles two common save formats:

    * **Full model** - ``torch.save(model, path)`` -> loaded and returned as-is.
    * **State-dict** - ``torch.save(model.state_dict(), path)`` -> raises
      ``ValueError`` with a helpful message, because the architecture is
      needed to reconstruct the model.

    Args:
        path: Filesystem path to the ``.pth`` / ``.pt`` file.

    Returns:
        PyTorch model in evaluation mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains a state-dict instead of a full model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

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
