"""Full-pipeline explainer base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .base_explainer import AbstractExplainer

if TYPE_CHECKING:
    from ..results import ConfiguredVisualiser, ExplanationResult


class FullExplainer(AbstractExplainer, ABC):
    """
    Explainer where you own the full ``explain`` pipeline end-to-end.

    Subclasses implement :meth:`explain` entirely — data conversion, model invocation,
    result construction, and artifact persistence.  Use this when the target library's
    API does not map to a simple ``compute_attributions(model, inputs) → Tensor`` step
    (e.g. Alibi Explain).
    """

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
        """Implement the full explanation pipeline."""
