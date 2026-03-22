"""Base class for attribution computation (framework-agnostic interface)"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from ..results import ExplanationResult, resolve_default_run_dir


class BaseExplainer(ABC):
    """
    Abstract base class for all explainer adapters.

    Subclasses wrap a specific framework (Captum, SHAP, …) and expose a
    unified ``compute_attributions`` method.  All orchestration (saving,
    visualisation) is handled by the top-level ``explain()`` function.
    """

    def __init__(self):
        self.attributions: torch.Tensor | None = None

    def explain(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        *,
        run_dir: str | Path | None = None,
        experiment_name: str | None = None,
        explainer_target: str | None = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        attributions = self.compute_attributions(model, inputs, **kwargs)
        self.attributions = attributions

        explanation = ExplanationResult(
            attributions=attributions,
            inputs=inputs,
            run_dir=Path(run_dir) if run_dir is not None else resolve_default_run_dir(),
            experiment_name=experiment_name,
            explainer_target=(explainer_target or f"{type(self).__module__}.{type(self).__name__}"),
            algorithm=getattr(self, "algorithm", ""),
            kwargs=kwargs,
        )
        explanation.write_artifacts()
        return explanation

    @abstractmethod
    def compute_attributions(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute attributions for the given inputs.

        Args:
            model:   PyTorch model to explain.
            inputs:  Input tensor (shape depends on modality).
            **kwargs: Framework-specific keyword arguments
                      (e.g. ``target``, ``baselines``, ``background_data``).

        Returns:
            Attribution tensor matching the input shape.
        """
