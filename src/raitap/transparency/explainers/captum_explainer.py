"""Captum explainer wrapper - handles ALL Captum attribution methods"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from raitap.transparency.algorithm_allowlist import ensure_algorithm_in_allowlist
from raitap.transparency.contracts import ExplanationPayloadKind, MethodFamily
from raitap.transparency.exceptions import ExplainerBackendIncompatibilityError
from raitap.transparency.explainers.registration import register_transparency_adapter

from .base_explainer import AttributionOnlyExplainer

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    import torch.nn as nn


@register_transparency_adapter(
    registry_name="captum",
    extra="captum",
    library="captum",
    # Captum emits the ``required_grads`` UserWarning on every run when inputs
    # don't already require gradients; it auto-fixes the issue so the warning
    # is pure noise. Scope ``module=`` to captum so unrelated UserWarnings
    # with matching messages aren't accidentally hidden.
    suppress_warnings=[(r"Input Tensor.*required_grads", UserWarning, r"captum.*")],
)
class CaptumExplainer(AttributionOnlyExplainer):
    """
    Single wrapper for ALL Captum attribution methods.

    Uses dynamic method loading - no need for class-per-method.
    """

    output_payload_kind: ClassVar[ExplanationPayloadKind] = ExplanationPayloadKind.ATTRIBUTIONS

    algorithm_registry: ClassVar[Mapping[str, frozenset[MethodFamily]]] = {
        "IntegratedGradients": frozenset({MethodFamily.GRADIENT}),
        "Saliency": frozenset({MethodFamily.GRADIENT}),
        "FeatureAblation": frozenset({MethodFamily.PERTURBATION}),
        "FeaturePermutation": frozenset({MethodFamily.PERTURBATION}),
        "Occlusion": frozenset({MethodFamily.PERTURBATION}),
        "ShapleyValueSampling": frozenset({MethodFamily.SHAPLEY, MethodFamily.PERTURBATION}),
        "ShapleyValues": frozenset({MethodFamily.SHAPLEY, MethodFamily.PERTURBATION}),
        "KernelShap": frozenset(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC}
        ),
        "Lime": frozenset(
            {MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC, MethodFamily.SURROGATE}
        ),
        "LayerGradCam": frozenset({MethodFamily.GRADIENT, MethodFamily.CAM}),
        "GuidedGradCam": frozenset({MethodFamily.GRADIENT, MethodFamily.CAM}),
    }

    ONNX_COMPATIBLE_ALGORITHMS: ClassVar[frozenset[str]] = frozenset(
        {
            "FeatureAblation",
            "FeaturePermutation",
            "Occlusion",
            "ShapleyValueSampling",
            "ShapleyValues",
            "KernelShap",
            "Lime",
        }
    )

    def __init__(self, algorithm: str, **init_kwargs):
        """
        Args:
            algorithm: Captum method name (e.g., "IntegratedGradients", "Saliency")
            **init_kwargs: Constructor arguments for the Captum method
                (from transparency YAML ``constructor:``), e.g. GradCAM ``layer=...``.
                Per-call options such as ``target`` belong under YAML ``call:``.
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
        del backend
        captum_attr = self._lazy_import("attr")

        # Dynamically get the method class
        try:
            method_class = getattr(captum_attr, self.algorithm)
        except AttributeError:
            raise ValueError(
                f"'{self.algorithm}' is not a valid captum.attr method.\n"
                f"Set algorithm to a class name available in captum.attr, "
                f"e.g. 'IntegratedGradients', 'Saliency', 'LayerGradCam'."
            ) from None

        init_kwargs = dict(self.init_kwargs)
        if self.algorithm in ("LayerGradCam", "GuidedGradCam"):
            layer_path = init_kwargs.pop("layer_path", None)
            if layer_path is not None and "layer" not in init_kwargs:
                init_kwargs["layer"] = _resolve_layer(model, str(layer_path))

        if self.algorithm == "Occlusion":
            attr_kwargs = _normalise_occlusion_kwargs(attr_kwargs)

        with self._rethrow():
            # Instantiate method with model and constructor args
            method = method_class(model, **init_kwargs)

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


def _resolve_layer(model: nn.Module, layer_path: str) -> nn.Module:
    layer: nn.Module = model
    for part in layer_path.split("."):
        try:
            layer = getattr(layer, part)
        except AttributeError as error:
            raise ValueError(f"Could not resolve layer_path {layer_path!r} on model.") from error
    return layer


def _normalise_occlusion_kwargs(attr_kwargs: dict[str, object]) -> dict[str, object]:
    normalised = dict(attr_kwargs)

    sliding_window_shapes = normalised.get("sliding_window_shapes")
    if isinstance(sliding_window_shapes, list):
        normalised["sliding_window_shapes"] = tuple(sliding_window_shapes)

    strides = normalised.get("strides")
    if isinstance(strides, list):
        normalised["strides"] = tuple(strides)

    return normalised
