"""SHAP explainer wrapper - handles ALL SHAP explainer types"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    MethodFamily,
)
from raitap.transparency.explainers.registration import transparency_adapter
from raitap.types import Capability

from .base_explainer import AttributionOnlyExplainer

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.models.access import ExplanationModel


# Which SHAP API each algorithm uses: legacy ``.shap_values()`` vs modern
# ``__call__ -> Explanation`` (masker-based). Separate from the semantics
# registry so dispatch mechanics don't bloat the families/stochastic data. (#267)
_SHAP_API: dict[str, str] = {
    "GradientExplainer": "legacy",
    "DeepExplainer": "legacy",
    "KernelExplainer": "legacy",
    "TreeExplainer": "legacy",
    "SamplingExplainer": "legacy",
    "PartitionExplainer": "modern",
    "ExactExplainer": "modern",
    "PermutationExplainer": "modern",
}


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
        ),
        "DeepExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.GRADIENT},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
            requires={Capability.AUTOGRAD},
        ),
        "KernelExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
            # Monte Carlo coalition sampling (np.random.choice / permutation).
            stochastic=True,
        ),
        # TreeExplainer is a tree-model method; requires=AUTOGRAD preserves current
        # gating, but it is the natural first consumer of the roadmap TREE_MODEL capability.
        "TreeExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.TREE},
            requires={Capability.AUTOGRAD},
        ),
        "SamplingExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
            stochastic=True,  # Monte Carlo sampling (run-twice non-deterministic)
        ),
        "PartitionExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
        ),
        "ExactExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
        ),
        "PermutationExplainer": ExplainerSemanticsHints(
            {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC},
            baseline_default=BaselineMode.INPUT_BATCH,
            baseline_cardinality=BaselineCardinality.SET,
            stochastic=True,  # random permutation order (seed=None default)
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
        """
        Compute SHAP values.

        Args:
            model: PyTorch model
            inputs: Input tensor
            input_spec: Inferred input specification; absorbed here so it does
                not leak into ``shap_values()``.
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
        del input_spec  # absorbed; must not reach shap_values()
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
