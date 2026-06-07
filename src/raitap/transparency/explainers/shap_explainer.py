"""SHAP explainer wrapper - handles ALL SHAP explainer types"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from raitap import raitap_log
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch
else:
    torch = lazy_import("torch")
from raitap.transparency.contracts import (
    BaselineCardinality,
    BaselineMode,
    ExplainerSemanticsHints,
    InputKind,
    MethodFamily,
)
from raitap.transparency.explainers.registration import transparency_adapter
from raitap.types import Capability

from .base_explainer import AttributionInvokeCtx, AttributionOnlyExplainer

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.models.access import ExplanationModel
    from raitap.transparency.contracts import InputSpec


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


def _modality_error(kind: str) -> ValueError:
    return ValueError(
        f"SHAP modern explainers support image and tabular inputs; got modality {kind!r}. "
        "Set transparency.<explainer>.raitap.input_metadata.kind to 'image' or 'tabular', "
        "or use a legacy SHAP explainer."
    )


def _normalise_modern_explanation(
    values: object,
    *,
    input_spec: InputSpec,
    target: int | list[int] | torch.Tensor | None,
) -> torch.Tensor:
    """Map ``Explanation.values`` to an input-shaped float32 attribution tensor.

    Modern shap returns class-last values: tabular ``(B, F, K)``, image NHWC+K
    ``(B, H, W, C, K)``. Cast to float32, select the target class via the shared
    helper, then (image, once the class axis is gone) permute NHWC -> NCHW to
    match RAITAP inputs. A class-selected image tensor is 4-D; with ``target`` of
    ``None`` the class axis is kept and no permute applies. (#267)
    """
    tensor = torch.as_tensor(np.asarray(values)).to(dtype=torch.float32)
    spec_shape = input_spec.shape
    inputs_ndim = len(spec_shape) if spec_shape is not None else tensor.ndim - 1
    selected = _select_target_attributions(tensor, inputs_ndim=inputs_ndim, target=target)
    if input_spec.kind is InputKind.IMAGE and selected.ndim == 4:
        # (B, H, W, C) -> (B, C, H, W)
        selected = selected.permute(0, 3, 1, 2).contiguous()
    return selected


def _build_masker(shap: Any, *, input_spec: InputSpec, background: object) -> Any:
    """Per-modality SHAP masker for the modern path. (#267)"""
    if input_spec.kind is InputKind.IMAGE:
        shape = input_spec.shape
        if shape is None or len(shape) != 4:
            raise _modality_error("image (need NCHW shape metadata)")
        _, c, h, w = shape  # NCHW
        return shap.maskers.Image("inpaint_telea", (h, w, c))
    if input_spec.kind is InputKind.TABULAR:
        return shap.maskers.Partition(_to_numpy(background))
    raise _modality_error(str(input_spec.kind))


def _modern_predict_fn(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    input_spec: InputSpec,
) -> Callable[[Any], Any]:
    """numpy->numpy predict fn for the modern masker path, fixing image layout."""
    is_image = input_spec.kind is InputKind.IMAGE

    def predict(masked: Any) -> Any:
        arr = np.asarray(masked, dtype="float32")
        tensor = torch.from_numpy(arr)
        if is_image and tensor.ndim == 4:
            tensor = tensor.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        out = model_fn(tensor)
        return out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else np.asarray(out)

    return predict


def _shap_legacy_invoker(ctx: AttributionInvokeCtx) -> torch.Tensor:
    """Invoke the legacy ``.shap_values()`` path for a SHAP registry entry (#266)."""
    ck = dict(ctx.call_kwargs)
    explainer = cast("ShapExplainer", ctx.explainer)
    return explainer._compute_legacy(
        ctx.library,
        ctx.model,
        ctx.inputs,
        background_data=ck.pop("background_data", None),
        target=ck.pop("target", None),
        input_spec=ctx.input_spec,
        **ck,
    )


def _shap_modern_invoker(ctx: AttributionInvokeCtx) -> torch.Tensor:
    """Invoke the modern masker-based path for a SHAP registry entry (#266)."""
    ck = dict(ctx.call_kwargs)
    explainer = cast("ShapExplainer", ctx.explainer)
    return explainer._compute_modern(
        ctx.library,
        ctx.model,
        ctx.inputs,
        background_data=ck.pop("background_data", None),
        target=ck.pop("target", None),
        input_spec=ctx.input_spec,
        **ck,
    )


@transparency_adapter(
    registry_name="shap",
    library="shap",
    error_patterns={
        (
            r"Output \d+ of BackwardHookFunctionBackward is a view "
            r"and is being modified inplace"
        ): (
            "DeepExplainer can fail on PyTorch models that use SiLU activations "
            "(for example EfficientNet variants) due to autograd/in-place "
            "limitations. Use alternatives like GradientExplainer."
        ),
    },
    # Gradient/Deep/Kernel fall back to the input batch when ``background_data``
    # is omitted; TreeExplainer takes no reference input (``baseline_default`` None).
    algorithm_registry={
        "GradientExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.GRADIENT},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
            requires={Capability.AUTOGRAD},
            # Samples random points along the path + background choice (rseed/nsamples).
            stochastic=True,
            invoker=_shap_legacy_invoker,
        ),
        "DeepExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.GRADIENT},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
            requires={Capability.AUTOGRAD},
            invoker=_shap_legacy_invoker,
        ),
        "KernelExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
            # Monte Carlo coalition sampling (np.random.choice / permutation).
            stochastic=True,
            invoker=_shap_legacy_invoker,
        ),
        # TreeExplainer is a tree-model method; requires=AUTOGRAD preserves current
        # gating, but it is the natural first consumer of the roadmap TREE_MODEL capability.
        "TreeExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.TREE},
            requires={Capability.AUTOGRAD},
            invoker=_shap_legacy_invoker,
        ),
        "SamplingExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
            stochastic=True,  # Monte Carlo sampling (run-twice non-deterministic)
            invoker=_shap_legacy_invoker,
        ),
        "PartitionExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
            invoker=_shap_modern_invoker,
        ),
        "ExactExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
            invoker=_shap_modern_invoker,
        ),
        "PermutationExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
            stochastic=True,  # random permutation order (seed=None default)
            invoker=_shap_modern_invoker,
        ),
    },
    baseline_kwarg_name="background_data",
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

    def compute_attributions(
        self,
        model: ExplanationModel,
        inputs: torch.Tensor,
        backend: object | None = None,
        input_spec: object | None = None,
        background_data: torch.Tensor | None = None,
        target: int | list[int] | torch.Tensor | None = None,
        **shap_kwargs,
    ) -> torch.Tensor:
        """Compute SHAP values via the per-entry invoker (legacy or modern).

        Args:
            model: PyTorch model
            inputs: Input tensor
            input_spec: Inferred input specification; passed to the invoker.
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
        hints = self.algorithm_registry.get(self.algorithm)
        ctx = AttributionInvokeCtx(
            explainer=self,
            library=shap,
            model=model,
            inputs=inputs,
            input_spec=cast("InputSpec | None", input_spec),
            call_kwargs={"background_data": background_data, "target": target, **shap_kwargs},
        )
        invoke = getattr(hints, "invoker", None) or _shap_legacy_invoker
        return invoke(ctx)

    def _compute_legacy(
        self,
        shap: Any,
        model: ExplanationModel,
        inputs: torch.Tensor,
        *,
        background_data: torch.Tensor | None,
        target: int | list[int] | torch.Tensor | None,
        input_spec: InputSpec | None,
        **shap_kwargs: Any,
    ) -> torch.Tensor:
        del input_spec  # absorbed; the legacy path does not use it

        # ---- legacy .shap_values() path ----
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

            if self.algorithm == "KernelExplainer":
                if not callable(model):
                    raise TypeError(
                        "KernelExplainer requires a callable (predict) model; "
                        f"got {type(model).__name__}."
                    )
                # model-agnostic: ``model`` is a predict callable; SHAP needs a
                # numpy fn. Wrap it (torch -> numpy) uniformly, regardless of backend.
                background_np = _to_numpy(background_data)
                explainer = explainer_class(
                    _make_backend_prediction_fn(model),
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

    def _compute_modern(
        self,
        shap: Any,
        model: ExplanationModel,
        inputs: torch.Tensor,
        *,
        background_data: torch.Tensor | None,
        target: int | list[int] | torch.Tensor | None,
        input_spec: InputSpec | None,
        **shap_kwargs: Any,
    ) -> torch.Tensor:
        if input_spec is None:
            raise _modality_error("unknown")
        if not callable(model):
            raise TypeError(
                f"{self.algorithm} (modern SHAP) requires a callable predict model; "
                f"got {type(model).__name__}."
            )
        background = background_data if background_data is not None else inputs
        masker = _build_masker(shap, input_spec=input_spec, background=background)
        predict = _modern_predict_fn(model, input_spec=input_spec)
        try:
            explainer_class = getattr(shap, self.algorithm)
        except AttributeError:
            raise ValueError(
                f"'{self.algorithm}' is not a valid shap explainer.\n"
                f"Set algorithm to a class name available in the shap package, "
                f"e.g. 'PartitionExplainer', 'ExactExplainer', 'PermutationExplainer'."
            ) from None
        explainer = explainer_class(predict, masker, **self.init_kwargs)

        inputs_np = inputs.detach().cpu().numpy()
        if input_spec.kind is InputKind.IMAGE:
            inputs_np = inputs_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC for the Image masker
        with self._rethrow():
            explanation = explainer(inputs_np, **shap_kwargs)
        return _normalise_modern_explanation(
            explanation.values,
            input_spec=input_spec,
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
