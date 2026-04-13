"""SHAP explainer wrapper - handles ALL SHAP explainer types"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn

from raitap.transparency.algorithm_allowlist import ensure_algorithm_in_allowlist
from raitap.transparency.contracts import ExplanationPayloadKind
from raitap.transparency.exceptions import ExplainerBackendIncompatibilityError

from .base_explainer import BaseExplainer

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ShapExplainer(BaseExplainer):
    """
    Single wrapper for ALL SHAP explainer types.

    Uses dynamic explainer loading - no need for class-per-explainer.
    """

    output_payload_kind: ClassVar[ExplanationPayloadKind] = ExplanationPayloadKind.ATTRIBUTIONS

    ONNX_COMPATIBLE_ALGORITHMS: frozenset[str] = frozenset({"KernelExplainer"})

    def __init__(self, algorithm: str, **init_kwargs):
        """
        Args:
            algorithm: SHAP explainer name (e.g., "GradientExplainer", "KernelExplainer")
            **init_kwargs: Constructor arguments for the SHAP explainer (from YAML ``constructor:``)
                Per-call options for ``shap_values`` belong under YAML ``call:``.
        """
        super().__init__()
        self.algorithm = algorithm
        self.init_kwargs = init_kwargs

    def check_backend_compat(self, backend: object) -> None:
        if getattr(backend, "supports_torch_autograd", False):
            return
        ensure_algorithm_in_allowlist(
            self.algorithm,
            type(self).ONNX_COMPATIBLE_ALGORITHMS,
            error_cls=ExplainerBackendIncompatibilityError,
            explainer=type(self).__name__,
            backend=type(backend).__name__,
            algorithm=self.algorithm,
            compatible_algorithms=sorted(type(self).ONNX_COMPATIBLE_ALGORITHMS),
        )

    def compute_attributions(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        backend: object | None = None,
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
            raise ImportError(
                "SHAP explainer is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

        # Dynamically get the explainer class
        try:
            explainer_class = getattr(shap, self.algorithm)
        except AttributeError:
            raise ValueError(
                f"'{self.algorithm}' is not a valid shap explainer.\n"
                f"Set algorithm to a class name available in the shap package, "
                f"e.g. 'GradientExplainer', 'DeepExplainer', 'KernelExplainer'."
            ) from None

        # Instantiate explainer (some need background data)
        # GradientExplainer, DeepExplainer, KernelExplainer REQUIRE background data
        if self.algorithm in ("GradientExplainer", "DeepExplainer", "KernelExplainer"):
            if background_data is None:
                logger.warning(
                    "%s: no background_data provided; using input batch as background "
                    "(results may be less accurate).",
                    self.algorithm,
                )
                background_data = inputs

            if (
                self.algorithm == "KernelExplainer"
                and backend is not None
                and not getattr(backend, "supports_torch_autograd", False)
            ):
                if not callable(backend):
                    raise TypeError(
                        "SHAP ONNX path requires a callable backend for KernelExplainer."
                    )
                callable_backend = cast("Callable[[torch.Tensor], torch.Tensor]", backend)
                background_np = _to_numpy(background_data)
                explainer = explainer_class(
                    _make_backend_prediction_fn(callable_backend),
                    background_np,
                    **self.init_kwargs,
                )
            else:
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


def _to_numpy(value: torch.Tensor | Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _make_backend_prediction_fn(
    backend: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[Any], Any]:
    def predict(inputs_np: Any) -> Any:
        tensor_inputs = torch.from_numpy(inputs_np)
        outputs = backend(tensor_inputs)
        if not isinstance(outputs, torch.Tensor):
            raise TypeError(
                "SHAP ONNX path expected backend predictions as torch.Tensor, "
                f"got {type(outputs).__name__}."
            )
        return outputs.detach().cpu().numpy()

    return predict
