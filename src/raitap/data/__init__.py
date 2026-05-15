"""
Data module — loading data from local files / URLs / demo samples, converting
to raw tensors, hosting the demo-sample list.

Heavy submodules (``data``, ``metadata``) and schema dataclasses are loaded
lazily via :pep:`562` ``__getattr__`` so that imports like
``from raitap.data.types import LabelEncoding`` — used by
``raitap.configs.schema`` during package init — do not re-enter
``raitap.configs`` mid-load and explode with a circular ImportError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import IdStrategy, LabelEncoding

if TYPE_CHECKING:
    from raitap.configs.schema import DataConfig, LabelsConfig

    from .data import Data, load_numpy_from_source, load_tensor_from_source
    from .metadata import DataInputMetadata, infer_data_input_metadata


__all__ = [
    "Data",
    "DataConfig",
    "DataInputMetadata",
    "IdStrategy",
    "LabelEncoding",
    "LabelsConfig",
    "infer_data_input_metadata",
    "load_numpy_from_source",
    "load_tensor_from_source",
]


# (attr name) -> (absolute module path, attr in that module)
_LAZY: dict[str, tuple[str, str]] = {
    "Data": ("raitap.data.data", "Data"),
    "load_numpy_from_source": ("raitap.data.data", "load_numpy_from_source"),
    "load_tensor_from_source": ("raitap.data.data", "load_tensor_from_source"),
    "DataInputMetadata": ("raitap.data.metadata", "DataInputMetadata"),
    "infer_data_input_metadata": ("raitap.data.metadata", "infer_data_input_metadata"),
    "DataConfig": ("raitap.configs.schema", "DataConfig"),
    "LabelsConfig": ("raitap.configs.schema", "LabelsConfig"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'raitap.data' has no attribute {name!r}")
    import importlib

    module_path, attr = target
    return getattr(importlib.import_module(module_path), attr)
