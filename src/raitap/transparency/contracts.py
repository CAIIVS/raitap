"""
Shared transparency contracts: payload kinds and explainer adapter typing.

:class:`ExplanationPayloadKind` labels the primary payload on
:class:`~raitap.transparency.results.ExplanationResult`.
:attr:`~ExplanationPayloadKind.ATTRIBUTIONS` is supported end-to-end (persistence,
visualisation, factory wiring). ``STRUCTURED`` labels a *principal* payload that is
not a standard attribution map (e.g. AIX360 prototypes / rule sets, #289). Additive
diagnostics such as convergence deltas attach via
``ExplanationResult.structured_payloads`` and do NOT change the principal
``payload_kind``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence  # noqa: TC003
from collections.abc import Set as AbstractSet  # noqa: TC003
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# Runtime import (not TYPE_CHECKING): ``Capability`` appears in the public
# ``ExplainerAlgorithmSpec.requires`` annotation, so ``typing.get_type_hints()``
# must resolve it from module globals. It is a torch-free StrEnum.
from raitap.types import Capability  # noqa: TC001

if TYPE_CHECKING:
    from raitap.reproducibility import Seeding

ConfiguredVisualiser = Any
ExplanationResult = Any
Module = Any
Tensor = Any


class ExplanationPayloadKind(StrEnum):
    """Primary payload category on ``ExplanationResult``."""

    ATTRIBUTIONS = "attributions"
    STRUCTURED = "structured"


class StructuredPayloadKind(StrEnum):
    """Typed kinds for additive structured payloads attached to an explanation.

    Additive sidecars on ``ExplanationResult.structured_payloads`` (independent of
    ``ExplanationPayloadKind``). ``CONVERGENCE_DELTA`` and ``BASE_VALUE`` are wired
    in #101; further kinds (contrastive examples, prototypes, rule sets) land with
    the AIX360 adapter (#289).
    """

    CONVERGENCE_DELTA = "convergence_delta"
    BASE_VALUE = "base_value"


class BaselineMode(StrEnum):
    """How an attribution baseline (IG ``baselines`` / SHAP ``background_data``) was obtained.

    Serialised by value into ``metadata.json`` and the report, so these string
    values are a stable contract.
    """

    CONFIGURED = "configured"  # resolved from a YAML data source (has provenance)
    USER_TENSOR = "user_tensor"  # tensor passed directly via the Python API
    ZERO = "zero"  # synthesized all-zeros (Captum's implicit default)
    INPUT_BATCH = "input_batch"  # the input batch (SHAP's implicit default)


class ExplanationScope(StrEnum):
    """Semantic breadth represented by an explanation artifact."""

    LOCAL = "local"
    AGGREGATED = "aggregated"
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
    DETECTION_BOXES = "detection_boxes"
    SEGMENTATION_MASK = "segmentation_mask"
    BBOX_REGRESSION = "bbox_regression"


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


class BaselineCardinality(StrEnum):
    """How many reference samples an algorithm's baseline is meant to hold.

    Used to validate (never reshape) a configured baseline against the method:

    * ``SINGLE`` — one reference input that broadcasts to the batch (Captum
      Integrated Gradients, DeepLift). A many-sample baseline is almost always a
      mistake (e.g. a SHAP background set reused here).
    * ``SET`` — a distribution of samples (SHAP ``background_data``); any count is
      meaningful, more is usually better.
    """

    SINGLE = "single"
    SET = "set"


@dataclass(frozen=True)
class StructuredPayload:
    """One additive payload attached to an ``ExplanationResult``.

    ``data`` is typed ``Any`` to stay general for #289 (prototypes, rule sets);
    #101 implements and tests the ``torch.Tensor`` branch only.
    """

    name: str
    kind: StructuredPayloadKind
    data: Any


@dataclass(frozen=True)
class StructuredOutputSpec:
    """Schema for one positional extra element of a tuple explainer output.

    Maps tuple position ``i + 1`` (position 0 is always the principal attribution).
    ``per_sample`` describes the intended batched-accumulation contract: a
    per-sample payload concatenates along dim 0 across chunks. #101 only ships
    per-sample payloads (convergence delta, SHAP base value) and the batched path
    always concatenates; the flag is a forward-declaration not yet read by code.
    Honouring ``per_sample=False`` (a global payload that must not be concatenated)
    is deferred to #289, which will add the guard when a non-per-sample payload
    first lands.
    """

    name: str
    kind: StructuredPayloadKind
    per_sample: bool = True


@dataclass(frozen=True)
class ExplainerAlgorithmSpec:
    """Per-algorithm metadata carried by a transparency adapter's ``algorithm_registry``.

    The transparency analogue of robustness's ``AssessorAlgorithmSpec``: one entry
    per algorithm an adapter wraps, holding everything the framework tracks and
    reports for that algorithm. ``baseline_default`` is the implicit
    ``BaselineMode`` used when the user omits the baseline kwarg (``None`` for
    algorithms with no meaningful default, e.g. Saliency / TreeExplainer).
    ``baseline_cardinality`` declares whether the baseline is a single reference or
    a sample set, so a mismatched ``raitap.baseline`` is flagged (not reshaped);
    ``None`` skips the check.
    """

    families: AbstractSet[MethodFamily]
    baseline_default: BaselineMode | None = None
    baseline_cardinality: BaselineCardinality | None = None
    requires: AbstractSet[Capability] = field(default_factory=frozenset)
    # RNG-source classification (issue #339). Replaces the old ``stochastic``
    # bool. ``deterministic`` => bit-reproducible; ``global_rng`` => covered by a
    # pinned global seed; ``self_seeded`` => owns a seed param, needs it passed.
    seeding: Seeding = "deterministic"
    # Optional per-algorithm invoker overriding the adapter's default
    # construct-and-call path (#266). None => default path.
    invoker: Any = None
    # Positional extra tuple outputs declared per algorithm (#101). Empty unless
    # the wrapped method appends diagnostics (e.g. IG return_convergence_delta).
    extra_outputs: tuple[StructuredOutputSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "families", frozenset(self.families))
        object.__setattr__(self, "requires", frozenset(self.requires))

    @property
    def stochastic(self) -> bool:
        """True when the algorithm depends on RNG (derived from ``seeding``)."""
        return self.seeding != "deterministic"


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
class DetectionBox:
    """Per-box metadata persisted with a detection explanation result.

    ``display_index`` is the rank in the filtered set (0..K-1) and is the
    user-facing ordinal. ``raw_index`` is the index in the clean forward
    pass's output list — useful for provenance and for re-anchoring the
    explanation to the original detection. ``xyxy`` is in input pixel space.
    """

    display_index: int
    raw_index: int
    xyxy: tuple[float, float, float, float]
    score: float
    label_index: int
    label_name: str | None = None
    # Ground-truth match (issue #233 Part 2). ``ground_truth_evaluated`` is True whenever
    # GT was available for this box's sample (so an unmatched box is a genuine
    # false positive, distinct from "no GT configured"). When matched,
    # ``true_label_index`` / ``true_label_name`` / ``true_match_iou`` describe the
    # highest-IoU GT box (class-agnostic match). All three stay None on a false
    # positive (ground_truth_evaluated True, no match) and when GT is absent.
    ground_truth_evaluated: bool = False
    true_label_index: int | None = None
    true_label_name: str | None = None
    true_match_iou: float | None = None


@dataclass(frozen=True)
class BaselineRecord:
    """Reference input an attribution method was computed against.

    Captured once at the explain chokepoint for methods that take a baseline
    (IG ``baselines``, SHAP ``background_data``), including implicit defaults.
    ``image_path`` is relative to the explanation ``run_dir`` and set only for
    image-modality runs; ``sha256`` hashes the tensor actually used as baseline.
    """

    kwarg_name: str
    mode: str
    source: str | None
    n_samples: int | None
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    image_path: Path | None


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
    method_families: AbstractSet[MethodFamily]
    target: ExplanationTarget | None
    sample_selection: SampleSelection | None
    input_spec: InputSpec | None
    output_space: OutputSpaceSpec
    # RNG-source classification (issue #339). Replaces the old ``stochastic``
    # bool. ``deterministic`` => bit-reproducible; ``global_rng`` => covered by a
    # pinned global seed; ``self_seeded`` => owns a seed param, needs it passed.
    seeding: Seeding = "deterministic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_families", frozenset(self.method_families))

    @property
    def stochastic(self) -> bool:
        """True when the result is RNG-dependent (derived from ``seeding``)."""
        return self.seeding != "deterministic"


@dataclass(frozen=True)
class ExplainerCapability:
    """Broad pre-compute semantic capability for an explainer configuration."""

    scope: ExplanationScope
    scope_definition_step: ScopeDefinitionStep
    payload_kind: ExplanationPayloadKind
    method_families: AbstractSet[MethodFamily]
    candidate_output_spaces: AbstractSet[ExplanationOutputSpace]

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_families", frozenset(self.method_families))
        object.__setattr__(self, "candidate_output_spaces", frozenset(self.candidate_output_spaces))


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
    detection_box: DetectionBox | None = None
    source_library: str | None = None
    method_families: AbstractSet[MethodFamily] = field(default_factory=frozenset)
    structured_payloads: tuple[StructuredPayload, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_families", frozenset(self.method_families))


@runtime_checkable
class ExplainerAdapter(Protocol):
    """
    Hydra explainer: ``explain`` matches :class:`~raitap.transparency.explainers.base_explainer.BaseExplainer`.

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
        output_root: str | Path | None = None,
        experiment_name: str | None = None,
        explainer_target: str | None = None,
        explainer_name: str | None = None,
        visualisers: list[ConfiguredVisualiser] | None = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        raise NotImplementedError
