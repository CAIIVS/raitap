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


@dataclass(frozen=True)
class ExplanationTarget:
    """Model output target described by an explanation."""

    target: int | str | Sequence[int] | None = None
    label: str | None = None


@dataclass(frozen=True)
class SampleSelection:
    sample_ids: list[str] | None
    sample_display_names: list[str] | None


@dataclass(frozen=True)
class InputSpec:
    """Input metadata used for deterministic semantic inference."""

    kind: str | None
    shape: tuple[int, ...] | None
    layout: str | None
    feature_names: list[str] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class OutputSpaceSpec:
    """Attribution output-space metadata."""

    space: ExplanationOutputSpace
    shape: tuple[int, ...] | None
    layout: str | None
    layer_path: str | None = None
    feature_names: list[str] | None = None
    requires_interpolation: bool = False


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
