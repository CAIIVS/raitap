"""SHAP explainer wrapper - handles ALL SHAP explainer types"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

from raitap import raitap_log
from raitap.transparency.algorithm_allowlist import ensure_algorithm_in_allowlist
from raitap.transparency.contracts import MethodFamily
from raitap.transparency.exceptions import ExplainerBackendIncompatibilityError
from raitap.transparency.explainers.registration import register_transparency_adapter

from .base_explainer import AttributionOnlyExplainer

if TYPE_CHECKING:
    from collections.abc import Callable


def _normalise_target_indices(
    target: list[int] | torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if isinstance(target, list):
        target_tensor = torch.tensor(target, device=device)
    else:
        target_tensor = target.to(device=device)

    target_tensor = target_tensor.to(dtype=torch.long).reshape(-1)
    if target_tensor.shape[0] != batch_size:
        raise ValueError(
            "Per-sample target must contain exactly one class index per input sample. "
            f"Expected {batch_size}, got {target_tensor.shape[0]}."
        )
    return target_tensor


def _select_target_attributions(
    shap_values: torch.Tensor,
    *,
    inputs_ndim: int,
    target: int | list[int] | torch.Tensor | None,
) -> torch.Tensor:
    if target is None or shap_values.ndim <= inputs_ndim:
        return shap_values

    if isinstance(target, int):
        return shap_values[..., target]

    target_tensor = _normalise_target_indices(
        target,
        device=shap_values.device,
        batch_size=shap_values.shape[0],
    )
    batch_indices = torch.arange(
        shap_values.shape[0],
        device=shap_values.device,
        dtype=torch.long,
    )
    return shap_values[batch_indices, ..., target_tensor]


@register_transparency_adapter(
    registry_name="shap",
    extra="shap",
    library="shap",
    error_patterns={
        re.compile(
            r"Output \d+ of BackwardHookFunctionBackward is a view "
            r"and is being modified inplace",
        ): (
            "DeepExplainer can fail on PyTorch models that use SiLU activations "
            "(for example EfficientNet variants) due to autograd/in-place "
            "limitations. Use alternatives like GradientExplainer."
        ),
    },
    algorithm_registry={
        "GradientExplainer": frozenset({MethodFamily.SHAPLEY, MethodFamily.GRADIENT}),
        "DeepExplainer": frozenset({MethodFamily.SHAPLEY, MethodFamily.GRADIENT}),
        "KernelExplainer": frozenset(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC}
        ),
        "TreeExplainer": frozenset({MethodFamily.SHAPLEY, MethodFamily.TREE}),
    },
    onnx_compatible_algorithms=frozenset({"KernelExplainer"}),
)
class ShapExplainer(AttributionOnlyExplainer):
    """
    Single wrapper for ALL SHAP explainer types.

    Uses dynamic explainer loading - no need for class-per-explainer.
    """

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
        shap = self._lazy_import()

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
                raitap_log.warn(
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
        with self._rethrow():
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

        return _select_target_attributions(
            shap_values,
            inputs_ndim=inputs.ndim,
            target=target,
        )


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
