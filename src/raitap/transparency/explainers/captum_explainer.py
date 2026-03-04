"""Captum explainer wrapper - handles ALL Captum attribution methods"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseExplainer


class CaptumExplainer(BaseExplainer):
    """
    Single wrapper for ALL Captum attribution methods.

    Uses dynamic method loading - no need for class-per-method.
    """

    def __init__(self, algorithm: str, **init_kwargs):
        """
        Args:
            algorithm: Captum method name (e.g., "IntegratedGradients", "Saliency")
            **init_kwargs: Constructor arguments for the Captum method
                - For GradCAM: layer=model.layer4
                - For most others: no args needed
        """
        super().__init__()
        self.algorithm = algorithm
        self.init_kwargs = init_kwargs

    def compute_attributions(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        target: int | list[int] | torch.Tensor | None = None,
        baselines: torch.Tensor | None = None,
        **attr_kwargs,
    ) -> torch.Tensor:
        """
        Compute Captum attributions.

        Args:
            model: PyTorch model
            inputs: Input tensor
            target: Target class index(es). Can be:
                - int: Same target for all samples
                - list[int]: Per-sample targets
                - torch.Tensor: Per-sample target tensor
            baselines: Baseline for integrated methods (optional)
            **attr_kwargs: Additional arguments for .attribute() method

        Returns:
            Attribution tensor matching input shape
        """
        try:
            import captum.attr
        except ImportError as e:
            raise ImportError(
                "Captum not installed. Install with: pip install captum>=0.7.0"
            ) from e

        # Dynamically get the method class
        try:
            method_class = getattr(captum.attr, self.algorithm)
        except AttributeError:
            raise ValueError(
                f"'{self.algorithm}' is not a valid captum.attr method.\n"
                f"Set algorithm to a class name available in captum.attr, "
                f"e.g. 'IntegratedGradients', 'Saliency', 'LayerGradCam'."
            ) from None

        # Instantiate method with model and constructor args
        method = method_class(model, **self.init_kwargs)

        # Compute attributions using unified Captum API
        # Only pass baselines if provided (some methods don't support it)
        if baselines is not None:
            attributions = method.attribute(
                inputs, target=target, baselines=baselines, **attr_kwargs
            )
        else:
            attributions = method.attribute(inputs, target=target, **attr_kwargs)

        # Captum already returns torch.Tensor, so just return
        return attributions
