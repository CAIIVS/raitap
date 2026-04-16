"""
Shared transparency contracts: payload kinds and explainer adapter typing.

:class:`ExplanationPayloadKind` labels the primary payload on
:class:`~raitap.transparency.results.ExplanationResult`.
:attr:`~ExplanationPayloadKind.ATTRIBUTIONS` is supported end-to-end (persistence,
visualisation, factory wiring). Other enum members may exist for forward-compatible
APIs before every code path is complete — for example
:attr:`~ExplanationPayloadKind.STRUCTURED` is not yet handled in
:meth:`~raitap.transparency.results.ExplanationResult.write_artifacts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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


@dataclass(frozen=True)
class VisualisationContext:
    """
    Standard RAITAP metadata provided to visualisers during the assessment pipeline.

    Encapsulates standard pipeline-controlled variables to avoid reflective
    signature inspection in the core logic.
    """

    algorithm: str
    sample_names: list[str] | None
    show_sample_names: bool


@runtime_checkable
class ExplainerAdapter(Protocol):
    """
    Hydra explainer: ``explain`` matches :class:`~raitap.transparency.explainers.base_explainer.AbstractExplainer`.

    Read ``output_payload_kind`` via :func:`raitap.transparency.contracts.explainer_output_kind`
    (not via direct attribute access — the attribute is optional and defaults to
    ``ATTRIBUTIONS`` when absent).
    """  # noqa: E501

    def check_backend_compat(self, backend: object) -> None:
        pass

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
    ) -> ExplanationResult:
        raise NotImplementedError
