"""SHAP explainer wrapper - handles ALL SHAP explainer types"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseExplainer


class ShapExplainer(BaseExplainer):
    """
    Single wrapper for ALL SHAP explainer types.

    Uses dynamic explainer loading - no need for class-per-explainer.
    """

    def __init__(self, algorithm: str, **init_kwargs):
        """
        Args:
            algorithm: SHAP explainer name (e.g., "GradientExplainer", "KernelExplainer")
            **init_kwargs: Constructor arguments for the SHAP explainer
        """
        super().__init__()
        self.algorithm = algorithm
        self.init_kwargs = init_kwargs

    def compute_attributions(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        background_data: torch.Tensor | None = None,
        target: int | list[int] | torch.Tensor | None = None,
        **shap_kwargs,
    ) -> torch.Tensor:
        """
        Compute SHAP values.

        Args:
            model: PyTorch model
            inputs: Input tensor
            background_data: Background dataset (REQUIRED for most explainers)
                - GradientExplainer: Required
                - DeepExplainer: Required
                - KernelExplainer: Required
                - TreeExplainer: Optional
            target: Target class(es) for attributions (optional)
                If not specified, returns attributions for all classes
            **shap_kwargs: Additional arguments for .shap_values() method

        Returns:
            SHAP values as torch.Tensor
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError("SHAP not installed. Install with: pip install shap>=0.46.0") from e

        # Dynamically get the explainer class
        try:
            explainer_class = getattr(shap, self.algorithm)
        except AttributeError:
            # Reference curated registry, not shap module
            from ..methods import SHAP

            available = [name for name in dir(SHAP) if not name.startswith("_")]
            raise ValueError(
                f"'{self.algorithm}' not in RAITAP's curated SHAP explainers.\n"
                f"Supported: {', '.join(available)}\n"
                f"To add: Test compatibility, then update transparency/methods.py"
            ) from None

        # Instantiate explainer (some need background data)
        # GradientExplainer, DeepExplainer, KernelExplainer REQUIRE background data
        if self.algorithm in ("GradientExplainer", "DeepExplainer", "KernelExplainer"):
            if background_data is None:
                raise ValueError(
                    f"{self.algorithm} requires background_data. "
                    f"Pass background_data to explain() method."
                )
            explainer = explainer_class(model, background_data, **self.init_kwargs)
        else:
            # TreeExplainer can work without background data
            if background_data is not None:
                explainer = explainer_class(model, background_data, **self.init_kwargs)
            else:
                explainer = explainer_class(model, **self.init_kwargs)

        # Compute SHAP values using unified SHAP API
        # GradientExplainer and DeepExplainer expect torch tensors
        # KernelExplainer and TreeExplainer expect numpy arrays
        if self.algorithm in ("GradientExplainer", "DeepExplainer"):
            # Keep as tensor for PyTorch-based explainers
            shap_values = explainer.shap_values(inputs, **shap_kwargs)
        else:
            # Convert to numpy for model-agnostic explainers
            inputs_np = inputs.cpu().numpy() if isinstance(inputs, torch.Tensor) else inputs
            shap_values = explainer.shap_values(inputs_np, **shap_kwargs)

        # Handle multi-class outputs: SHAP returns list of arrays for each class
        # or a single array with shape (*input_shape, num_classes)
        if isinstance(shap_values, list):
            # List of arrays, one per class - stack them
            shap_values = torch.stack(
                [
                    torch.from_numpy(v) if not isinstance(v, torch.Tensor) else v
                    for v in shap_values
                ],
                dim=-1,
            )
        elif isinstance(shap_values, torch.Tensor):
            # Already a tensor, keep it
            pass
        else:
            # Numpy array
            shap_values = torch.from_numpy(shap_values)

        # If target specified and we have per-class attributions, select target class
        if target is not None and shap_values.ndim > inputs.ndim:
            # shap_values has an extra dimension for classes
            # Select the target class for each sample
            if isinstance(target, int):
                # Same target for all samples
                shap_values = shap_values[..., target]
            else:
                # Per-sample targets
                if isinstance(target, list):
                    target = torch.tensor(target)
                # Select using advanced indexing
                batch_indices = torch.arange(shap_values.shape[0])
                shap_values = shap_values[batch_indices, ..., target]

        return shap_values
