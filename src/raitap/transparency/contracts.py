"""
Shared transparency contracts: payload kinds and explainer adapter typing.

:class:`ExplanationPayloadKind` labels the primary payload on
:class:`~raitap.transparency.results.ExplanationResult`.
:attr:`~ExplanationPayloadKind.ATTRIBUTIONS` is supported end-to-end (persistence,
visualisation, factory wiring). Other enum members may exist for forward-compatible
APIs before every code path is complete — for example
:attr:`~ExplanationPayloadKind.STRUCTURED` is not yet handled in
:meth:`~raitap.transparency.results.ExplanationResult.write_artifacts`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence  # noqa: TC003
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import Any, Protocol, runtime_checkable

ConfiguredVisualiser = Any
ExplanationResult = Any
Module = Any
Tensor = Any


class ExplanationPayloadKind(StrEnum):
    """Primary payload category on ``ExplanationResult``."""

    ATTRIBUTIONS = "attributions"
    STRUCTURED = "structured"


class ExplanationScope(StrEnum):
    """Semantic breadth represented by an explanation artifact."""

    LOCAL = "local"
    COHORT = "cohort"
    GLOBAL = "global"


class ScopeDefinitionStep(StrEnum):
    """Pipeline step that defined an artifact's semantic scope."""

    EXPLAINER_OUTPUT = "explainer_output"
    VISUALISER_SUMMARY = "visualiser_summary"


class ExplanationOutputSpace(StrEnum):
    """Coordinate space represented by attribution values."""

    INPUT_FEATURES = "input_features"
    INTERPRETABLE_FEATURES = "interpretable_features"
    LAYER_ACTIVATION = "layer_activation"
    IMAGE_SPATIAL_MAP = "image_spatial_map"
    TOKEN_SEQUENCE = "token_sequence"


class InputKind(StrEnum):
    """Known input modalities used by semantic validation."""

    IMAGE = "image"
    TABULAR = "tabular"
    TEXT = "text"
    TIME_SERIES = "time_series"


class TensorLayout(StrEnum):
    """Named tensor layouts accepted by built-in visualisers."""

    BATCH_CHANNEL_HEIGHT_WIDTH = "NCHW"
    BATCH_FEATURE = "(B,F)"
    BATCH_TIME_CHANNEL = "(B,T,C)"
    TOKEN_SEQUENCE = "TOKENS"


class MethodFamily(StrEnum):
    """Explicit algorithm families supported by the typed transparency contract."""

    GRADIENT = "gradient"
    PERTURBATION = "perturbation"
    SHAPLEY = "shapley"
    CAM = "cam"
    MODEL_AGNOSTIC = "model_agnostic"
    TREE = "tree"
    SURROGATE = "surrogate"


def explainer_output_kind(explainer: object) -> ExplanationPayloadKind:
    raw = getattr(type(explainer), "output_payload_kind", None)
    if isinstance(raw, ExplanationPayloadKind):
        return raw
    return ExplanationPayloadKind.ATTRIBUTIONS


def explainer_output_scope(explainer: object) -> ExplanationScope:
    raw = getattr(type(explainer), "output_scope", None)
    if isinstance(raw, ExplanationScope):
        return raw
    return ExplanationScope.LOCAL


@dataclass(frozen=True)
class ExplanationTarget:
    """Model output target described by an explanation."""

    target: int | str | Sequence[int] | None = None
    label: str | None = None


@dataclass(frozen=True)
class SampleSelection:
    sample_ids: list[str] | None
    sample_display_names: list[str] | None


@dataclass(frozen=True, init=False)
class InputSpec:
    """Input metadata used for deterministic semantic inference."""

    kind: InputKind | None
    shape: tuple[int, ...] | None
    layout: TensorLayout | None
    feature_names: list[str] | None = None
    metadata: Mapping[str, Any] | None = None

    def __init__(
        self,
        kind: InputKind | str | None,
        shape: tuple[int, ...] | None,
        layout: TensorLayout | str | None,
        feature_names: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "kind", normalise_input_kind(kind))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "layout", normalise_tensor_layout(layout))
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, init=False)
class OutputSpaceSpec:
    """Attribution output-space metadata."""

    space: ExplanationOutputSpace
    shape: tuple[int, ...] | None
    layout: TensorLayout | None
    layer_path: str | None = None
    feature_names: list[str] | None = None
    requires_interpolation: bool = False

    def __init__(
        self,
        space: ExplanationOutputSpace,
        shape: tuple[int, ...] | None,
        layout: TensorLayout | str | None,
        layer_path: str | None = None,
        feature_names: list[str] | None = None,
        requires_interpolation: bool = False,
    ) -> None:
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "layout", normalise_tensor_layout(layout))
        object.__setattr__(self, "layer_path", layer_path)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "requires_interpolation", requires_interpolation)


@dataclass(frozen=True)
class VisualSummarySpec:
    """Metadata for visualisers that summarize a set of local explanations."""

    summary_type: str
    aggregation: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ExplanationSemantics:
    """Typed contract describing the meaning of an explanation artifact."""

    scope: ExplanationScope
    scope_definition_step: ScopeDefinitionStep
    payload_kind: ExplanationPayloadKind
    method_families: frozenset[MethodFamily]
    target: ExplanationTarget | None
    sample_selection: SampleSelection | None
    input_spec: InputSpec | None
    output_space: OutputSpaceSpec


@dataclass(frozen=True)
class ExplainerCapability:
    """Broad pre-compute semantic capability for an explainer configuration."""

    scope: ExplanationScope
    scope_definition_step: ScopeDefinitionStep
    payload_kind: ExplanationPayloadKind
    method_families: frozenset[MethodFamily]
    candidate_output_spaces: frozenset[ExplanationOutputSpace]


def normalise_input_kind(value: InputKind | str | None) -> InputKind | None:
    """Return a typed input kind from a string or enum boundary value."""

    if value is None or isinstance(value, InputKind):
        return value
    raw = str(value).strip().lower()
    if raw == "timeseries":
        return InputKind.TIME_SERIES
    for candidate in InputKind:
        if raw in {candidate.value, candidate.name.lower()}:
            return candidate
    raise ValueError(f"Unknown input kind {value!r}.")


def normalise_tensor_layout(value: TensorLayout | str | None) -> TensorLayout | None:
    """Return a typed tensor layout from a string or enum boundary value."""

    if value is None or isinstance(value, TensorLayout):
        return value
    raw = str(value).strip()
    compact = raw.upper().replace(" ", "")
    aliases = {
        TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH.value: TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
        TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH.name: TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
        "B,F": TensorLayout.BATCH_FEATURE,
        TensorLayout.BATCH_FEATURE.value: TensorLayout.BATCH_FEATURE,
        TensorLayout.BATCH_FEATURE.name: TensorLayout.BATCH_FEATURE,
        "B,T,C": TensorLayout.BATCH_TIME_CHANNEL,
        TensorLayout.BATCH_TIME_CHANNEL.value: TensorLayout.BATCH_TIME_CHANNEL,
        TensorLayout.BATCH_TIME_CHANNEL.name: TensorLayout.BATCH_TIME_CHANNEL,
        "TOKEN_SEQUENCE": TensorLayout.TOKEN_SEQUENCE,
        TensorLayout.TOKEN_SEQUENCE.value: TensorLayout.TOKEN_SEQUENCE,
        TensorLayout.TOKEN_SEQUENCE.name: TensorLayout.TOKEN_SEQUENCE,
    }
    try:
        return aliases[compact]
    except KeyError as error:
        raise ValueError(f"Unknown tensor layout {value!r}.") from error


@dataclass(frozen=True)
class VisualisationContext:
    """
    Standard RAITAP metadata provided to visualisers during the assessment pipeline.

    Encapsulates standard pipeline-controlled variables to avoid reflective
    signature inspection in the core logic.
    """

    algorithm: str
    sample_names: list[str] | None
    show_sample_names: bool


@runtime_checkable
class ExplainerAdapter(Protocol):
    """
    Hydra explainer: ``explain`` matches :class:`~raitap.transparency.explainers.base_explainer.AbstractExplainer`.

    Read ``output_payload_kind`` via :func:`raitap.transparency.contracts.explainer_output_kind`
    (not via direct attribute access — the attribute is optional and defaults to
    ``ATTRIBUTIONS`` when absent).
    """  # noqa: E501

    def check_backend_compat(self, backend: object) -> None:
        pass

    def explain(
        self,
        model: Module,
        inputs: Tensor,
        *,
        backend: object | None = None,
        run_dir: str | Path | None = None,
        output_root: str | Path = ".",
        experiment_name: str | None = None,
        explainer_target: str | None = None,
        explainer_name: str | None = None,
        visualisers: list[ConfiguredVisualiser] | None = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        raise NotImplementedError
