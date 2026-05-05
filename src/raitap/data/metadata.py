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


def is_image_source(source: str) -> bool:
    if source in SAMPLE_SOURCES:
        return True
    path = Path(source)
    if path.suffix.lower() in _IMAGE_EXTENSIONS:
        return True
    if path.is_dir():
        return any(child.suffix.lower() in _IMAGE_EXTENSIONS for child in path.iterdir())
    return False


def is_tabular_source(source: str) -> bool:
    path = Path(source)
    if path.suffix.lower() in _TABULAR_EXTENSIONS:
        return True
    if path.is_dir():
        return any(child.suffix.lower() in _TABULAR_EXTENSIONS for child in path.iterdir())
    return False


def shape_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        return tuple(int(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
