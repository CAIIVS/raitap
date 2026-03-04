"""Base class for attribution computation (framework-agnostic interface)"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseExplainer(ABC):
    """
    Abstract base class for all explainer adapters.

    Subclasses wrap a specific framework (Captum, SHAP, …) and expose a
    unified ``compute_attributions`` method.  All orchestration (saving,
    visualisation) is handled by the top-level ``explain()`` function.
    """

    def __init__(self):
        self.attributions: torch.Tensor | None = None

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
