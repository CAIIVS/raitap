from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class FormatConverter(Protocol):
    """Convert a model file of any supported format to a ``.pth`` file path."""

    def convert(self, path: Path) -> Path:
        """Convert *path* and return the path to the resulting ``.pth`` file."""
        ...


class PthConverter:
    """No-op converter for native ``.pth`` / ``.pt`` files."""

    def convert(self, path: Path) -> Path:
        return path


CONVERTERS: dict[str, FormatConverter] = {
    ".pth": PthConverter(),
    ".pt": PthConverter(),
}
