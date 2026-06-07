"""Captum explainer wrapper - handles ALL Captum attribution methods"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from raitap.transparency.contracts import (
    BaselineCardinality,
    BaselineMode,
    ExplainerAlgorithmSpec,
    MethodFamily,
)
from raitap.transparency.explainers.registration import transparency_adapter
from raitap.types import Capability

from .base_explainer import AttributionOnlyExplainer

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from raitap.models.access import ExplanationModel


@transparency_adapter(
    registry_name="captum",
    library="captum",
    # Captum emits the ``required_grads`` UserWarning on every run when inputs
    # don't already require gradients; it auto-fixes the issue so the warning
    # is pure noise. Scope ``module=`` to captum so unrelated UserWarnings
    # with matching messages aren't accidentally hidden.
    suppress_warnings=[(r"Input Tensor.*required_grads", UserWarning, r"captum.*")],
    # Integral methods that fall back to a zero reference when ``baselines`` is
    # omitted (IntegratedGradients, LayerIntegratedGradients, LayerConductance,
    # LayerDeepLift) carry ``baseline_default=BaselineMode.ZERO``; methods that
    # take no reference input leave it ``None``.
    algorithm_registry={
        "IntegratedGradients": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            baseline_default=BaselineMode.ZERO,
            baseline_cardinality=BaselineCardinality.SINGLE,
            requires={Capability.AUTOGRAD},
        ),
        "Saliency": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            requires={Capability.AUTOGRAD},
        ),
        "FeatureAblation": ExplainerAlgorithmSpec({MethodFamily.PERTURBATION}),
        "FeaturePermutation": ExplainerAlgorithmSpec({MethodFamily.PERTURBATION}),
        "Occlusion": ExplainerAlgorithmSpec({MethodFamily.PERTURBATION}),
        "ShapleyValueSampling": ExplainerAlgorithmSpec(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION},
            stochastic=True,  # random permutation sampling
        ),
        "ShapleyValues": ExplainerAlgorithmSpec({MethodFamily.SHAPLEY, MethodFamily.PERTURBATION}),
        "KernelShap": ExplainerAlgorithmSpec(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            stochastic=True,  # random coalition sampling
        ),
        "Lime": ExplainerAlgorithmSpec(
            {MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC, MethodFamily.SURROGATE},
            stochastic=True,  # random perturbation sampling
        ),
        "LayerGradCam": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT, MethodFamily.CAM},
            requires={Capability.AUTOGRAD},
        ),
        "GuidedGradCam": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT, MethodFamily.CAM},
            requires={Capability.AUTOGRAD},
        ),
        "LayerConductance": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            baseline_default=BaselineMode.ZERO,
            baseline_cardinality=BaselineCardinality.SINGLE,
            requires={Capability.AUTOGRAD},
        ),
        "LayerIntegratedGradients": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            baseline_default=BaselineMode.ZERO,
            baseline_cardinality=BaselineCardinality.SINGLE,
            requires={Capability.AUTOGRAD},
        ),
        "LayerActivation": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            requires={Capability.AUTOGRAD},
        ),
        "LayerDeepLift": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            baseline_default=BaselineMode.ZERO,
            baseline_cardinality=BaselineCardinality.SINGLE,
            requires={Capability.AUTOGRAD},
        ),
        "LayerGradientXActivation": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            requires={Capability.AUTOGRAD},
        ),
        "LayerLRP": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            requires={Capability.AUTOGRAD},
        ),
    },
    baseline_kwarg_name="baselines",
)
class CaptumExplainer(AttributionOnlyExplainer):
    """
    Single wrapper for ALL Captum attribution methods.

    Uses dynamic method loading - no need for class-per-method.
    """

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

    def compute_attributions(
        self,
        model: ExplanationModel,
        inputs: torch.Tensor,
        backend: object | None = None,
        input_spec: object | None = None,
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
        del backend, input_spec
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
        if _needs_layer_resolution(self.algorithm):
            layer_path = init_kwargs.pop("layer_path", None)
            if layer_path is not None and "layer" not in init_kwargs:
                # Layer*/GuidedGradCam require AUTOGRAD, so ``model`` is always a
                # live ``nn.Module`` here (never a predict callable).
                init_kwargs["layer"] = _resolve_layer(cast("nn.Module", model), str(layer_path))

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


def _needs_layer_resolution(algorithm: str) -> bool:
    """Whether an algorithm takes a captum ``layer`` constructor argument.

    True for every ``Layer*`` method plus ``GuidedGradCam``. ``Neuron*`` methods
    also take ``layer`` but are out of scope here (they need a call-time neuron
    selector too); see #269. Drives ``layer_path -> layer`` resolution. (#267)
    """
    return algorithm.startswith("Layer") or algorithm == "GuidedGradCam"


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
