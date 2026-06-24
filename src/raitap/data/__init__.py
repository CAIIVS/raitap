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

from .types import IdStrategy, LabelEncoding, Preprocessing

if TYPE_CHECKING:
    from raitap.configs.schema import DataConfig, LabelsConfig

    from .data import Data, load_numpy_from_source, load_tensor_from_source
    from .label_formats import LabelFormatAdapter, resolve_label_format_adapter
    from .metadata import DataInputMetadata, infer_data_input_metadata
    from .preprocessing import (
        DataPreprocessingFactory,
        ModelInputTransformationFactory,
        raitap_model_input_transformation_factory,
        raitap_preprocessing_factory,
    )


__all__ = [
    "Data",
    "DataConfig",
    "DataInputMetadata",
    "DataPreprocessingFactory",
    "IdStrategy",
    "LabelEncoding",
    "LabelFormatAdapter",
    "LabelsConfig",
    "ModelInputTransformationFactory",
    "Preprocessing",
    "infer_data_input_metadata",
    "load_numpy_from_source",
    "load_tensor_from_source",
    "raitap_model_input_transformation_factory",
    "raitap_preprocessing_factory",
    "resolve_label_format_adapter",
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
    "DataPreprocessingFactory": ("raitap.data.preprocessing", "DataPreprocessingFactory"),
    "ModelInputTransformationFactory": (
        "raitap.data.preprocessing",
        "ModelInputTransformationFactory",
    ),
    "raitap_model_input_transformation_factory": (
        "raitap.data.preprocessing",
        "raitap_model_input_transformation_factory",
    ),
    "raitap_preprocessing_factory": (
        "raitap.data.preprocessing",
        "raitap_preprocessing_factory",
    ),
    "LabelFormatAdapter": ("raitap.data.label_formats", "LabelFormatAdapter"),
    "resolve_label_format_adapter": (
        "raitap.data.label_formats",
        "resolve_label_format_adapter",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'raitap.data' has no attribute {name!r}")
    import importlib

    module_path, attr = target
    return getattr(importlib.import_module(module_path), attr)
