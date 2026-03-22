from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class FormatConverter(Protocol):
    """Load a model file of any supported format and return the model."""

    def convert(self, path: Path) -> nn.Module:
        """Load *path* and return the model as an ``nn.Module``."""
        ...


class PthConverter:
    """Loader for native ``.pth`` / ``.pt`` files."""

    def convert(self, path: Path) -> nn.Module:
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


CONVERTERS: dict[str, FormatConverter] = {
    ".pth": PthConverter(),
    ".pt": PthConverter(),
}
