from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torchvision import models

from .converters import CONVERTERS

if TYPE_CHECKING:
    import torch.nn as nn

    from raitap.configs.schema import AppConfig
    from raitap.tracking import BaseTracker


class Model:
    def __init__(self, config: AppConfig) -> None:
        self.network = self._load_model(config)

    def _load_model(self, config: AppConfig) -> nn.Module:
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
            f"Supported file formats: {list(CONVERTERS)}\n"
            f"To use your own model, set source to a valid model file path."
        )

    def log(self, tracker: BaseTracker) -> None:
        tracker.log_model(self.network)


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

    return converter.convert(path)


def _load_pretrained(model_name: str) -> nn.Module:
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
    return model
