"""Bridge between ``raitap.data`` and the transparency/robustness phases:
infer the :class:`InputSpec` to thread through explainers + assessors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raitap.data import infer_data_input_metadata
from raitap.transparency.contracts import InputSpec

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.data import Data


def input_metadata_for_data(config: AppConfig, data: Data) -> InputSpec | None:
    """Return runtime input metadata derived from the data object, or ``None``
    if neither ``kind`` nor ``layout`` can be determined (in which case any
    ``transparency.<explainer>.raitap.input_metadata`` from the explainer
    config will be used unchanged)."""
    explicit = getattr(data, "input_metadata", None)
    if isinstance(explicit, InputSpec):
        return explicit
    config_explicit = config.data.input_metadata
    if isinstance(config_explicit, InputSpec):
        return config_explicit
    metadata = infer_data_input_metadata(config, data)
    if metadata.kind is None and metadata.layout is None:
        # Don't override yaml-provided ``raitap.input_metadata`` with an empty
        # spec — let the explainer-level config drive output-space inference.
        return None
    return InputSpec(
        kind=metadata.kind,
        shape=metadata.shape,
        layout=metadata.layout,
        feature_names=metadata.feature_names,
        metadata=metadata.metadata,
    )
