from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torchvision import models

from .backend import ModelBackend, OnnxBackend, TorchBackend

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.tracking import BaseTracker


class Model:
    def __init__(self, config: AppConfig) -> None:
        self.backend = self._load_model(config)

    def _load_model(self, config: AppConfig) -> ModelBackend:
        source = config.model.source
        if not source:
            raise ValueError(
                "No model specified. Set model.source in your config.\n"
                "  model.source: path/to/your_model.pth   (custom model)\n"
                "  model.source: resnet50                 (built-in demo model)"
            )

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
            f"Supported file formats: {_supported_model_formats()}\n"
            f"To use your own model, set source to a valid model file path."
        )

    def log(self, tracker: BaseTracker) -> None:
        tracker.log_model(self.backend)


def _load_from_path(path: Path) -> ModelBackend:
    """
    Load a model backend from a file path.

    Args:
        path: Path to the model file.

    Returns:
        Model backend ready for inference and explanation.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is unsupported.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix.lower() == ".onnx":
        return OnnxBackend.from_path(path)

    if path.suffix.lower() in {".pth", ".pt"}:
        return TorchBackend(_load_torch_module_from_path(path))

    raise ValueError(
        f"Unsupported model format {path.suffix!r}. Supported formats: {_supported_model_formats()}"
    )


def _load_torch_module_from_path(path: Path) -> nn.Module:
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


def _load_pretrained(model_name: str) -> ModelBackend:
    """
    Load a torchvision model with its default pre-trained weights.

    This is intended for demos and quick testing.  For production use, supply
    a model file path via :class:`Model` instead.

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

    model = factory(weights="DEFAULT")
    model.eval()
    return TorchBackend(model)


def _supported_model_formats() -> list[str]:
    return [".onnx", ".pt", ".pth"]
