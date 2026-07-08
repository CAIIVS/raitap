"""Captum explainer wrapper - handles ALL Captum attribution methods"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from raitap.task_families.base import ATTENTION_MASK_KEY
from raitap.transparency.contracts import (
    BaselineCardinality,
    BaselineMode,
    ExplainerAlgorithmSpec,
    InputKind,
    MethodFamily,
    StructuredOutputSpec,
    StructuredPayloadKind,
)
from raitap.transparency.explainers.registration import transparency_adapter
from raitap.types import Capability

from .base_explainer import AttributionInvokeCtx, AttributionOnlyExplainer

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from raitap.models.access import ExplanationModel
    from raitap.transparency.contracts import InputSpec


_CONVERGENCE_DELTA = (
    StructuredOutputSpec("convergence_delta", StructuredPayloadKind.CONVERGENCE_DELTA),
)


def _noise_tunnel_base_choices() -> set[str]:
    """Valid ``base_algorithm`` names: non-layer, gradient-family captum methods.

    Derived from the registry so it never drifts. Today: {Saliency,
    IntegratedGradients}; auto-expands as gradient methods are added.
    """
    registry = CaptumExplainer.algorithm_registry
    return {
        name
        for name, spec in registry.items()
        if name != "NoiseTunnel"
        and not _needs_layer_resolution(name)
        and MethodFamily.GRADIENT in spec.families
    }


def _resolve_noise_tunnel_base(explainer: CaptumExplainer) -> type:
    """Validate ``base_algorithm`` and return its captum class.

    The base must be a vetted non-layer gradient method (see
    ``_noise_tunnel_base_choices``) so NoiseTunnel's static GRADIENT/AUTOGRAD
    registry entry and input-feature output-space inference stay correct.
    """
    base_name = explainer.init_kwargs.get("base_algorithm")
    valid = _noise_tunnel_base_choices()
    choices = sorted(valid)
    if base_name is None:
        raise ValueError(
            "NoiseTunnel requires a 'base_algorithm' constructor kwarg naming the "
            "captum method to wrap, e.g. constructor: { base_algorithm: Saliency }. "
            f"Valid options: {choices}."
        )
    if base_name not in valid:
        raise ValueError(
            f"NoiseTunnel base_algorithm={base_name!r} is not supported. Choose a "
            f"non-layer, gradient-family captum method. Valid options: {choices}."
        )
    captum_attr = explainer._lazy_import("attr")
    return cast("type", getattr(captum_attr, base_name))


def _noise_tunnel_invoker(ctx: AttributionInvokeCtx) -> torch.Tensor:
    """Build ``NoiseTunnel(base(model))`` and run SmoothGrad/VarGrad (#269)."""
    explainer = cast("CaptumExplainer", ctx.explainer)
    captum_attr = ctx.library
    call_kwargs = dict(ctx.call_kwargs)
    target = call_kwargs.pop("target", None)
    # baselines (if any) ride **call_kwargs to the base method; drop the key when
    # None so captum does not see baselines=None.
    if call_kwargs.get("baselines") is None:
        call_kwargs.pop("baselines", None)

    base_class = _resolve_noise_tunnel_base(explainer)
    with explainer._rethrow():
        base = base_class(ctx.model)
        noise_tunnel = captum_attr.NoiseTunnel(base)
        return noise_tunnel.attribute(ctx.inputs, target=target, **call_kwargs)


@transparency_adapter(
    registry_name="captum",
    import_name="captum",
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
            extra_outputs=_CONVERGENCE_DELTA,
        ),
        "Saliency": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            requires={Capability.AUTOGRAD},
        ),
        "NoiseTunnel": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            requires={Capability.AUTOGRAD},
            # Adds Gaussian noise (smoothgrad/vargrad); run-twice non-deterministic.
            seeding="global_rng",
            invoker=_noise_tunnel_invoker,
        ),
        "FeatureAblation": ExplainerAlgorithmSpec({MethodFamily.PERTURBATION}),
        "FeaturePermutation": ExplainerAlgorithmSpec({MethodFamily.PERTURBATION}),
        "Occlusion": ExplainerAlgorithmSpec({MethodFamily.PERTURBATION}),
        "ShapleyValueSampling": ExplainerAlgorithmSpec(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION},
            seeding="global_rng",  # random permutation sampling
        ),
        "ShapleyValues": ExplainerAlgorithmSpec({MethodFamily.SHAPLEY, MethodFamily.PERTURBATION}),
        "KernelShap": ExplainerAlgorithmSpec(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            seeding="global_rng",  # random coalition sampling
        ),
        "Lime": ExplainerAlgorithmSpec(
            {MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC, MethodFamily.SURROGATE},
            seeding="global_rng",  # random perturbation sampling
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
            extra_outputs=_CONVERGENCE_DELTA,
        ),
        "LayerIntegratedGradients": ExplainerAlgorithmSpec(
            {MethodFamily.GRADIENT},
            baseline_default=BaselineMode.ZERO,
            baseline_cardinality=BaselineCardinality.SINGLE,
            requires={Capability.AUTOGRAD},
            extra_outputs=_CONVERGENCE_DELTA,
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
            extra_outputs=_CONVERGENCE_DELTA,
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
        """Compute Captum attributions via the per-entry invoker (#266).

        Default methods use ``_default_captum_invoker`` (the uniform
        ``method_class(model).attribute(...)`` path). Methods with a non-uniform
        lifecycle (NoiseTunnel) carry a custom ``invoker`` on their registry entry.
        """
        del backend
        captum_attr = self._lazy_import("attr")
        hints = self.algorithm_registry.get(self.algorithm)
        ctx = AttributionInvokeCtx(
            explainer=self,
            library=captum_attr,
            model=model,
            inputs=inputs,
            input_spec=cast("InputSpec | None", input_spec),
            call_kwargs={"target": target, "baselines": baselines, **attr_kwargs},
        )
        invoke = getattr(hints, "invoker", None) or _default_captum_invoker
        return invoke(ctx)


def _default_captum_invoker(ctx: AttributionInvokeCtx) -> torch.Tensor:
    """Uniform Captum path: ``method_class(model, **init).attribute(inputs, ...)``."""
    explainer = cast("CaptumExplainer", ctx.explainer)
    captum_attr = ctx.library
    call_kwargs = dict(ctx.call_kwargs)
    target = call_kwargs.pop("target", None)
    baselines = call_kwargs.pop("baselines", None)
    # ``attention_mask`` (e.g. from a tokenized text input) is not attributed or
    # perturbed like a regular input feature; Captum passes it through untouched
    # to the forward as a positional ``additional_forward_args`` entry (#340).
    attention_mask = call_kwargs.pop(ATTENTION_MASK_KEY, None)
    # Merge, do not clobber: pop any caller-provided ``additional_forward_args``
    # too and append the mask, so a config that already sets forward args keeps
    # them AND the mask is applied (a plain spread would let one override the
    # other and silently drop the mask). (#340)
    caller_forward_args = call_kwargs.pop("additional_forward_args", None)
    forward_args: tuple[Any, ...] = ()
    if caller_forward_args is not None:
        forward_args += (
            caller_forward_args
            if isinstance(caller_forward_args, tuple)
            else (caller_forward_args,)
        )
    if attention_mask is not None:
        forward_args += (attention_mask,)
    extra_call_kwargs = {"additional_forward_args": forward_args} if forward_args else {}

    try:
        method_class = getattr(captum_attr, explainer.algorithm)
    except AttributeError:
        raise ValueError(
            f"'{explainer.algorithm}' is not a valid captum.attr method.\n"
            f"Set algorithm to a class name available in captum.attr, "
            f"e.g. 'IntegratedGradients', 'Saliency', 'LayerGradCam'."
        ) from None

    init_kwargs = dict(explainer.init_kwargs)
    if _needs_layer_resolution(explainer.algorithm):
        layer_path = init_kwargs.pop("layer_path", None)
        if layer_path is not None and "layer" not in init_kwargs:
            init_kwargs["layer"] = _resolve_layer(cast("nn.Module", ctx.model), str(layer_path))

    if explainer.algorithm == "Occlusion":
        call_kwargs = _normalise_occlusion_kwargs(call_kwargs)

    # Text models (HF ``AutoModelForSequenceClassification``) return a
    # ``SequenceClassifierOutput`` object, not a bare tensor; Captum needs the
    # forward to yield a tensor. Wrap the forward to reduce it to logits. The
    # ``layer`` above is resolved on the real ``ctx.model`` so LayerIntegratedGradients
    # hooks fire on the same module the wrapper calls. Image/tabular inputs pass
    # ``ctx.model`` through unchanged (byte-identical). (#340)
    forward_func = _forward_func_for(ctx)

    with explainer._rethrow():
        method = method_class(forward_func, **init_kwargs)
        if baselines is not None:
            attributions = method.attribute(
                ctx.inputs,
                target=target,
                baselines=baselines,
                **extra_call_kwargs,
                **call_kwargs,
            )
        else:
            attributions = method.attribute(
                ctx.inputs, target=target, **extra_call_kwargs, **call_kwargs
            )
    return _reduce_text_embedding_dim(attributions, ctx)


def _reduce_text_embedding_dim(attributions: Any, ctx: AttributionInvokeCtx) -> Any:
    """Collapse the embedding axis of text attributions to per-token scores.

    Layer attribution over a text embedding layer returns ``(B, T, H)`` (one
    vector per token). The canonical Captum text convention sums over the
    embedding dimension ``H`` to get a single ``(B, T)`` importance per token,
    which the ``TOKEN_SEQUENCE`` output space + ``CaptumTextVisualiser`` consume.
    Non-text inputs (and already-reduced outputs) are returned unchanged. (#340)
    """
    kind = getattr(ctx.input_spec, "kind", None)
    if kind is InputKind.TEXT and getattr(attributions, "ndim", 0) == 3:
        return attributions.sum(dim=-1)
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


def _forward_func_for(ctx: AttributionInvokeCtx) -> Any:
    """Return the callable Captum should attribute through.

    Text models emit an object output (HF ``SequenceClassifierOutput``); Captum
    requires a tensor, so wrap the forward to reduce it to the primary tensor
    (logits). Non-text inputs return ``ctx.model`` unchanged so the image/tabular
    paths stay byte-identical. (#340)
    """
    model = ctx.model
    kind = getattr(ctx.input_spec, "kind", None)
    if kind is not InputKind.TEXT:
        return model

    from raitap.pipeline.phases.forward_pass import extract_primary_tensor

    def _logits_forward(*forward_args: Any) -> Any:
        return extract_primary_tensor(model(*forward_args))

    return _logits_forward


def _normalise_occlusion_kwargs(attr_kwargs: dict[str, object]) -> dict[str, object]:
    normalised = dict(attr_kwargs)

    sliding_window_shapes = normalised.get("sliding_window_shapes")
    if isinstance(sliding_window_shapes, list):
        normalised["sliding_window_shapes"] = tuple(sliding_window_shapes)

    strides = normalised.get("strides")
    if isinstance(strides, list):
        normalised["strides"] = tuple(strides)

    return normalised
