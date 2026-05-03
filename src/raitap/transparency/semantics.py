"""Semantic registry helpers for transparency explainers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    ExplainerCapability,
    ExplanationOutputSpace,
    ExplanationScope,
    InputSpec,
    MethodFamily,
    OutputSpaceSpec,
    ScopeDefinitionStep,
    explainer_output_kind,
)

SHAP_METHOD_FAMILIES: Mapping[str, frozenset[MethodFamily]] = {
    "GradientExplainer": frozenset({MethodFamily.SHAPLEY, MethodFamily.GRADIENT}),
    "DeepExplainer": frozenset({MethodFamily.SHAPLEY, MethodFamily.GRADIENT}),
    "KernelExplainer": frozenset(
        {MethodFamily.SHAPLEY, MethodFamily.PERTURBATION, MethodFamily.MODEL_AGNOSTIC}
    ),
    "TreeExplainer": frozenset({MethodFamily.SHAPLEY, MethodFamily.TREE}),
}

CAPTUM_METHOD_FAMILIES: Mapping[str, frozenset[MethodFamily]] = {
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


def method_families_for_explainer(explainer: object) -> frozenset[MethodFamily]:
    """Return the explicit method-family mapping for a configured explainer."""

    framework, algorithm = _framework_and_algorithm(explainer)
    registry = _registry_for_framework(framework)
    families = registry.get(algorithm)
    if families is None:
        raise ValueError(_method_family_error(framework, algorithm))
    return families


def explainer_capability(explainer: object) -> ExplainerCapability:
    """Return broad pre-compute semantic capabilities for an explainer."""

    method_families = method_families_for_explainer(explainer)
    return ExplainerCapability(
        scope=ExplanationScope.LOCAL,
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=explainer_output_kind(explainer),
        method_families=method_families,
        candidate_output_spaces=_candidate_output_spaces(method_families),
    )


def infer_input_spec(
    inputs: object | None = None,
    *,
    input_metadata: InputSpec | Mapping[str, Any] | None = None,
    kind: str | None = None,
    layout: str | None = None,
    feature_names: Sequence[str] | None = None,
) -> InputSpec:
    """Build an ``InputSpec`` from explicit metadata and tensor shape."""

    if isinstance(input_metadata, InputSpec):
        return input_metadata

    metadata = dict(input_metadata) if input_metadata is not None else {}
    resolved_kind = kind if kind is not None else _optional_str(metadata.get("kind"))
    resolved_layout = layout if layout is not None else _optional_str(metadata.get("layout"))
    resolved_features = (
        list(feature_names)
        if feature_names is not None
        else _optional_str_list(metadata.get("feature_names"))
    )

    shape = _shape_tuple(metadata.get("shape"))
    if shape is None:
        shape = _shape_tuple(getattr(inputs, "shape", None))

    return InputSpec(
        kind=resolved_kind,
        shape=shape,
        layout=resolved_layout,
        feature_names=resolved_features,
        metadata=metadata or None,
    )


def infer_output_space(
    *,
    input_spec: InputSpec,
    attributions: object | None = None,
    explainer: object | None = None,
    algorithm: str | None = None,
    method_families: frozenset[MethodFamily] | None = None,
    layer_path: str | None = None,
    feature_names: Sequence[str] | None = None,
) -> OutputSpaceSpec:
    """Infer deterministic output-space metadata from input and method semantics."""

    resolved_method_families = _resolve_method_families(
        method_families=method_families,
        explainer=explainer,
        algorithm=algorithm,
    )

    shape = _shape_tuple(getattr(attributions, "shape", None))
    features = list(feature_names) if feature_names is not None else input_spec.feature_names

    input_kind = (input_spec.kind or "").lower()
    input_layout = (input_spec.layout or "").upper()
    if MethodFamily.CAM in resolved_method_families:
        if input_kind == "image" or input_layout == "NCHW":
            return OutputSpaceSpec(
                space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP,
                shape=shape,
                layout=input_spec.layout,
                layer_path=layer_path,
                feature_names=features,
                requires_interpolation=True,
            )
        return OutputSpaceSpec(
            space=ExplanationOutputSpace.LAYER_ACTIVATION,
            shape=shape,
            layout=input_spec.layout,
            layer_path=layer_path,
            feature_names=features,
            requires_interpolation=False,
        )

    if input_kind == "text" or input_layout in {"TOKENS", "TOKEN_SEQUENCE"}:
        return OutputSpaceSpec(
            space=ExplanationOutputSpace.TOKEN_SEQUENCE,
            shape=shape,
            layout=input_spec.layout,
            feature_names=features,
        )

    if input_kind in {"time_series", "timeseries"}:
        return OutputSpaceSpec(
            space=ExplanationOutputSpace.INPUT_FEATURES,
            shape=shape,
            layout=input_spec.layout,
            layer_path=layer_path,
            feature_names=features,
        )

    if input_kind in {"image", "tabular"} or input_layout in {"NCHW", "B,F", "(B,F)"}:
        return OutputSpaceSpec(
            space=ExplanationOutputSpace.INPUT_FEATURES,
            shape=shape,
            layout=input_spec.layout,
            layer_path=layer_path,
            feature_names=features,
        )

    raise ValueError(
        "Output-space inference requires explicit input metadata; shape alone is ambiguous."
    )


def _framework_and_algorithm(explainer: object) -> tuple[str, str]:
    raw_algorithm = getattr(explainer, "algorithm", None)
    if raw_algorithm is None:
        raise ValueError(_method_family_error(type(explainer).__name__, "<missing>"))

    algorithm = str(raw_algorithm)
    raw_framework = getattr(explainer, "framework", None)
    if raw_framework is None:
        raw_framework = getattr(explainer, "explainer_framework", None)
    if raw_framework is not None:
        return _normalise_framework(str(raw_framework)), algorithm

    class_name = type(explainer).__name__
    module = type(explainer).__module__.lower()
    if class_name == "ShapExplainer" or "shap" in module:
        return "SHAP", algorithm
    if class_name == "CaptumExplainer" or "captum" in module:
        return "Captum", algorithm

    if algorithm in SHAP_METHOD_FAMILIES and algorithm not in CAPTUM_METHOD_FAMILIES:
        return "SHAP", algorithm
    if algorithm in CAPTUM_METHOD_FAMILIES and algorithm not in SHAP_METHOD_FAMILIES:
        return "Captum", algorithm

    return class_name, algorithm


def _resolve_method_families(
    *,
    method_families: frozenset[MethodFamily] | None,
    explainer: object | None,
    algorithm: str | None,
) -> frozenset[MethodFamily]:
    if method_families is not None:
        return method_families
    if algorithm is not None:
        return _method_families_for_algorithm(algorithm)
    if explainer is not None:
        return method_families_for_explainer(explainer)
    return frozenset()


def _method_families_for_algorithm(algorithm: str) -> frozenset[MethodFamily]:
    matches: list[tuple[str, frozenset[MethodFamily]]] = []
    if algorithm in SHAP_METHOD_FAMILIES:
        matches.append(("SHAP", SHAP_METHOD_FAMILIES[algorithm]))
    if algorithm in CAPTUM_METHOD_FAMILIES:
        matches.append(("Captum", CAPTUM_METHOD_FAMILIES[algorithm]))

    if len(matches) == 1:
        return matches[0][1]
    if not matches:
        raise ValueError(_method_family_error("<unknown>", algorithm))

    frameworks = ", ".join(framework for framework, _families in matches)
    raise ValueError(
        "method-family inference is ambiguous for algorithm "
        f"{algorithm}; matched frameworks {frameworks}."
    )


def _registry_for_framework(framework: str) -> Mapping[str, frozenset[MethodFamily]]:
    if framework == "SHAP":
        return SHAP_METHOD_FAMILIES
    if framework == "Captum":
        return CAPTUM_METHOD_FAMILIES
    return {}


def _normalise_framework(framework: str) -> str:
    lowered = framework.strip().lower()
    if lowered == "shap":
        return "SHAP"
    if lowered == "captum":
        return "Captum"
    return framework


def _method_family_error(framework: str, algorithm: str) -> str:
    return (
        "method-family inference is not implemented for framework "
        f"{framework} and algorithm {algorithm}."
    )


def _candidate_output_spaces(
    method_families: frozenset[MethodFamily],
) -> frozenset[ExplanationOutputSpace]:
    if MethodFamily.CAM in method_families:
        return frozenset(
            {
                ExplanationOutputSpace.IMAGE_SPATIAL_MAP,
                ExplanationOutputSpace.LAYER_ACTIVATION,
            }
        )
    return frozenset(
        {
            ExplanationOutputSpace.INPUT_FEATURES,
            ExplanationOutputSpace.INTERPRETABLE_FEATURES,
            ExplanationOutputSpace.TOKEN_SEQUENCE,
        }
    )


def _shape_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        return tuple(int(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_str_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]
