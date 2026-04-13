"""
Shared transparency contracts: payload kinds and explainer adapter typing.

New :class:`ExplanationPayloadKind` values are added only when a feature implements
persistence, visualisation, and factory checks for that kind end-to-end.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from .results import ConfiguredVisualiser, ExplanationResult


class ExplanationPayloadKind(StrEnum):
    """Primary payload category on ``ExplanationResult``."""

    ATTRIBUTIONS = "attributions"
    STRUCTURED = "structured"


def explainer_output_kind(explainer: object) -> ExplanationPayloadKind:
    raw = getattr(type(explainer), "output_payload_kind", None)
    if isinstance(raw, ExplanationPayloadKind):
        return raw
    return ExplanationPayloadKind.ATTRIBUTIONS


@runtime_checkable
class ExplainerAdapter(Protocol):
    """
    Hydra explainer: ``explain`` matches ``BaseExplainer``.

    Read ``output_payload_kind`` via :func:`explainer_output_kind` (not via Protocol fields).
    """

    output_payload_kind: ClassVar[ExplanationPayloadKind]

    def check_backend_compat(self, backend: object) -> None: ...

    def explain(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        *,
        backend: object | None = None,
        run_dir: str | Path | None = None,
        output_root: str | Path = ".",
        experiment_name: str | None = None,
        explainer_target: str | None = None,
        explainer_name: str | None = None,
        visualisers: list[ConfiguredVisualiser] | None = None,
        **kwargs: Any,
    ) -> ExplanationResult: ...
