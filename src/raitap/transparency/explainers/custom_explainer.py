"""Base class for third-party explainer adapters that do not use ``compute_attributions``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from raitap.transparency.contracts import ExplanationPayloadKind

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from raitap.transparency.results import ConfiguredVisualiser, ExplanationResult


class CustomExplainer(ABC):
    """
    Explainer that implements the full ``explain`` pipeline (no ``compute_attributions``).

    Use for libraries whose APIs do not map to tensor-in/tensor-out attribution (e.g. Alibi).
    """

    output_payload_kind: ClassVar[ExplanationPayloadKind] = ExplanationPayloadKind.ATTRIBUTIONS

    def check_backend_compat(self, backend: object) -> None:
        del backend
        return None

    @abstractmethod
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
        """Match :meth:`~raitap.transparency.explainers.base_explainer.BaseExplainer.explain`."""
        ...
