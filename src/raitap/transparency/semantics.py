"""Semantic registry helpers for transparency explainers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from raitap.reproducibility import (
    Seeding,  # noqa: TC001  must stay runtime-resolvable for get_type_hints()
)
from raitap.types import TaskKind  # noqa: TC001  must stay runtime-resolvable for get_type_hints()
from raitap.utils.diagnostics import Diagnostic, Module
from raitap.utils.errors import RaitapError

from .contracts import (
    ExplainerAlgorithmSpec,
    ExplainerCapability,
    ExplanationOutputSpace,
    InputKind,
    InputSpec,
    MethodFamily,
    OutputSpaceSpec,
    ScopeDefinitionStep,
    TensorLayout,
    explainer_output_kind,
    explainer_output_scope,
)


def method_families_for_explainer(explainer: object) -> frozenset[MethodFamily]:
    """Return the explicit method-family mapping for a configured explainer.

    Reads the adapter's own ``algorithm_registry`` ClassVar — a required
    class-body attribute on every transparency adapter, validated at decoration
    time by ``@adapters.transparency``. Mirrors ``robustness.hints_for_assessor``.
    """
    algorithm = _explainer_algorithm(explainer)
    registry = getattr(type(explainer), "algorithm_registry", None)
    if isinstance(registry, Mapping) and algorithm in registry:
        return registry[algorithm].families
    raise _method_family_error(type(explainer).__name__, algorithm)


def explainer_seeding(explainer: object) -> Seeding:
    """Resolve the configured explainer's RNG classification (issue #339).

    Reads the adapter's own ``algorithm_registry``; never raises.
    Any unresolved case (stubs, unknown algorithm) defaults to ``deterministic`` —
    a missing declaration must never over-claim stochasticity.
    """
    hints = _hints_for_explainer(explainer)
    return hints.seeding if hints is not None else "deterministic"


def _hints_for_explainer(explainer: object) -> ExplainerAlgorithmSpec | None:
    """Resolve the registry hints for a configured explainer, or ``None``."""
    try:
        algorithm = _explainer_algorithm(explainer)
    except RaitapError:
        return None
    registry = getattr(type(explainer), "algorithm_registry", None)
    if isinstance(registry, Mapping) and algorithm in registry:
        return registry[algorithm]
    return None


def _explainer_algorithm(explainer: object) -> str:
    raw = getattr(explainer, "algorithm", None)
    if raw is None:
        raise _method_family_error(type(explainer).__name__, "<missing>")
    return str(raw)


def explainer_capability(
    explainer: object, *, task_kind: TaskKind | None = None
) -> ExplainerCapability:
    """Return broad pre-compute semantic capabilities for an explainer."""

    method_families = method_families_for_explainer(explainer)
    return ExplainerCapability(
        scope=explainer_output_scope(explainer),
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=explainer_output_kind(explainer),
        method_families=method_families,
        candidate_output_spaces=_candidate_output_spaces(method_families, task_kind=task_kind),
    )


def infer_input_spec(
    inputs: object | None = None,
    *,
    input_metadata: InputSpec | Mapping[str, Any] | None = None,
    kind: InputKind | str | None = None,
    layout: TensorLayout | str | None = None,
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
    method_families: frozenset[MethodFamily] | None = None,
    layer_path: str | None = None,
    feature_names: Sequence[str] | None = None,
    task_kind: TaskKind | None = None,
) -> OutputSpaceSpec:
    """Infer deterministic output-space metadata from input and method semantics."""

    if task_kind is not None:
        from raitap.task_families import resolve_task_family

        fixed = resolve_task_family(task_kind).fixed_output_space
        if fixed is not None:
            shape = _shape_tuple(getattr(attributions, "shape", None))
            features = (
                list(feature_names) if feature_names is not None else input_spec.feature_names
            )
            return OutputSpaceSpec(
                space=fixed,
                shape=shape,
                layout=input_spec.layout,
                layer_path=layer_path,
                feature_names=features,
            )

    resolved_method_families = _resolve_method_families(
        method_families=method_families,
        explainer=explainer,
    )

    shape = _shape_tuple(getattr(attributions, "shape", None))
    features = list(feature_names) if feature_names is not None else input_spec.feature_names

    input_kind = input_spec.kind
    input_layout = input_spec.layout
    if MethodFamily.CAM in resolved_method_families:
        if input_kind is InputKind.IMAGE or input_layout is TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH:
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

    # Text token attribution is token-aligned even via a Layer* method
    # (LayerIntegratedGradients over the embeddings is the canonical text path):
    # the embedding axis is collapsed to a per-token score, so the result lives
    # in TOKEN_SEQUENCE space, not opaque layer space. Text therefore takes
    # precedence over the generic ``layer_path`` branch below. (#340)
    if input_kind is InputKind.TEXT or input_layout is TensorLayout.TOKEN_SEQUENCE:
        return OutputSpaceSpec(
            space=ExplanationOutputSpace.TOKEN_SEQUENCE,
            shape=shape,
            layout=input_spec.layout,
            feature_names=features,
        )

    if layer_path is not None:
        # Non-CAM Layer* attribution is layer-space, not input-aligned; tag
        # LAYER_ACTIVATION (skips INPUT_FEATURES shape validation). (#267)
        return OutputSpaceSpec(
            space=ExplanationOutputSpace.LAYER_ACTIVATION,
            shape=shape,
            layout=input_spec.layout,
            layer_path=layer_path,
            feature_names=features,
            requires_interpolation=False,
        )

    if input_kind is InputKind.TIME_SERIES or input_layout is TensorLayout.BATCH_TIME_CHANNEL:
        return OutputSpaceSpec(
            space=ExplanationOutputSpace.INPUT_FEATURES,
            shape=shape,
            layout=input_spec.layout,
            layer_path=layer_path,
            feature_names=features,
        )

    if input_kind in {InputKind.IMAGE, InputKind.TABULAR} or input_layout in {
        TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
        TensorLayout.BATCH_FEATURE,
    }:
        return OutputSpaceSpec(
            space=ExplanationOutputSpace.INPUT_FEATURES,
            shape=shape,
            layout=input_spec.layout,
            layer_path=layer_path,
            feature_names=features,
        )

    raise ValueError(
        "Output-space inference requires explicit input metadata; shape alone is "
        "ambiguous. Set either ``input_metadata.kind`` "
        "(one of: image, tabular, text, time_series) or ``input_metadata.layout`` "
        "(one of: NCHW, (B,F), (B,T,C), TOKENS) — a recognised value for either "
        "key is enough to disambiguate. Direct callers pass via "
        "``infer_input_spec(input_metadata=...)``; Hydra users set "
        "``transparency.<explainer>.raitap.input_metadata`` in config. "
        "Example::\n\n"
        "    raitap:\n"
        "      input_metadata:\n"
        "        kind: image\n"
        "        layout: NCHW\n"
    )


def _resolve_method_families(
    *,
    method_families: frozenset[MethodFamily] | None,
    explainer: object | None,
) -> frozenset[MethodFamily]:
    if method_families is not None:
        return method_families
    if explainer is not None:
        return method_families_for_explainer(explainer)
    return frozenset()


def _method_family_error(explainer_name: str, algorithm: str) -> RaitapError:
    if algorithm == "<missing>":
        message = (
            f"Explainer {explainer_name} has no `algorithm` set. Add "
            "`transparency.<name>.algorithm: ...` to your config (e.g. "
            "`GradientExplainer` for SHAP, `IntegratedGradients` for Captum)."
        )
    else:
        message = (
            f"Unknown algorithm {algorithm!r} for explainer {explainer_name}: "
            "not in its algorithm_registry."
        )
    return RaitapError(
        message,
        diagnostic=Diagnostic(
            module=Module.transparency,
            file=__file__,
            line=0,
            third_party_lib=None,
        ),
    )


def _candidate_output_spaces(
    method_families: frozenset[MethodFamily],
    task_kind: TaskKind | None = None,
) -> frozenset[ExplanationOutputSpace]:
    if task_kind is not None:
        from raitap.task_families import resolve_task_family

        fixed = resolve_task_family(task_kind).fixed_output_space
        if fixed is not None:
            return frozenset({fixed})
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
