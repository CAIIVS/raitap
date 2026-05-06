from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raitap.configs import cfg_to_dict

from .samples import SAMPLE_SOURCES

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_TABULAR_EXTENSIONS = {".csv", ".tsv", ".parquet"}


@dataclass(frozen=True)
class DataInputMetadata:
    kind: str | None
    shape: tuple[int, ...] | None
    layout: str | None
    feature_names: list[str] | None = None
    metadata: dict[str, Any] | None = None


def infer_data_input_metadata(config: object, data: object) -> DataInputMetadata:
    explicit = getattr(data, "input_metadata", None)
    if explicit is None:
        explicit = getattr(getattr(config, "data", None), "input_metadata", None)
    if explicit is not None:
        explicit_mapping = cfg_to_dict(explicit)
        kind = explicit_mapping.get("kind")
        layout = explicit_mapping.get("layout")
        feature_names = explicit_mapping.get("feature_names")
        shape = explicit_mapping.get("shape", getattr(getattr(data, "tensor", None), "shape", None))
        return DataInputMetadata(
            kind=None if kind is None else str(kind),
            shape=shape_tuple(shape),
            layout=None if layout is None else str(layout),
            feature_names=None if feature_names is None else [str(item) for item in feature_names],
            metadata=dict(explicit_mapping),
        )

    source = str(
        getattr(data, "source", None) or getattr(getattr(config, "data", None), "source", "")
    )
    shape = shape_tuple(getattr(getattr(data, "tensor", None), "shape", None))
    if is_image_source(source):
        return DataInputMetadata(kind="image", shape=shape, layout="NCHW")
    if is_tabular_source(source):
        return DataInputMetadata(kind="tabular", shape=shape, layout="(B,F)")
    return DataInputMetadata(kind=None, shape=shape, layout=None)


def _case_insensitive_glob(ext: str) -> str:
    """Build a case-insensitive glob suffix for ``ext`` (e.g. ``.jpg`` →
    ``*.[jJ][pP][gG]``)."""
    return "*" + "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in ext)


def _has_extension_recursive(path: Path, extensions: set[str]) -> bool:
    """True if any file under ``path`` (recursively) has a matching extension.

    Case-insensitive: ``IMG_001.JPG`` matches ``.jpg``. Iterates per-extension
    case-insensitive globs so the OS layer skips non-matching dirents instead
    of yielding every file. Short-circuits on the first match. Filesystem
    errors (unreadable subdirs, broken symlinks, etc.) are swallowed — this
    helper is best-effort source detection, not data validation.
    """
    try:
        for ext in sorted(extensions):
            if next(path.rglob(_case_insensitive_glob(ext)), None) is not None:
                return True
    except OSError:
        return False
    return False


def is_image_source(source: str) -> bool:
    if source in SAMPLE_SOURCES:
        return True
    path = Path(source)
    if path.suffix.lower() in _IMAGE_EXTENSIONS:
        return True
    if path.is_dir():
        return _has_extension_recursive(path, _IMAGE_EXTENSIONS)
    return False


def is_tabular_source(source: str) -> bool:
    path = Path(source)
    if path.suffix.lower() in _TABULAR_EXTENSIONS:
        return True
    if path.is_dir():
        return _has_extension_recursive(path, _TABULAR_EXTENSIONS)
    return False


def shape_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        return tuple(int(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
