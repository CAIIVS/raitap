---
title: "Types"
description: "Reference for the key public dataclasses, enums, and protocols exported by RAITAP."
---

RAITAP is a Python package, so its type surface is expressed through dataclasses, enums, and protocols rather than TypeScript interfaces. The most important public types are defined in `src/raitap/configs/schema.py`, `src/raitap/transparency/contracts.py`, `src/raitap/run/outputs.py`, `src/raitap/metrics/base_metric.py`, and `src/raitap/reporting/sections.py`.

Typical import paths:

```python
from raitap.configs import AppConfig
from raitap.data import DataInputMetadata
from raitap.metrics import MetricResult
from raitap.reporting import ReportGroup, ReportManifest, ReportSection
from raitap.run import PredictionSummary, RunOutputs
from raitap.transparency import (
    ExplainerCapability,
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    ExplanationSemantics,
    ExplanationTarget,
    InputKind,
    InputSpec,
    MethodFamily,
    OutputSpaceSpec,
    SampleSelection,
    ScopeDefinitionStep,
    TensorLayout,
    VisualSummarySpec,
)
```

## Config dataclasses

Source: `src/raitap/configs/schema.py`

```python
@dataclass
class ModelConfig:
    source: str | None = None
    arch: str | None = None
    num_classes: int | None = None
    pretrained: bool = False

@dataclass
class LabelsConfig:
    source: str | None = None
    id_column: str | None = None
    column: str | None = None
    encoding: str | None = None
    id_strategy: str = "auto"

@dataclass
class DataConfig:
    name: str = "isic2018"
    description: str | None = None
    source: str | None = None
    forward_batch_size: int | None = None
    labels: LabelsConfig = field(default_factory=LabelsConfig)

@dataclass
class TransparencyConfig:
    _target_: str = "CaptumExplainer"
    algorithm: str = "IntegratedGradients"
    constructor: dict[str, Any] = field(default_factory=dict)
    call: dict[str, Any] = field(default_factory=dict)
    raitap: dict[str, Any] = field(default_factory=dict)
    visualisers: list[Any] = field(default_factory=lambda: [{"_target_": "CaptumImageVisualiser"}])
```

These types define the stable public shape of RAITAP configs. They matter because factories and helpers assume these fields exist even when framework-specific nested content is loosely typed.

## Transparency enums

Source: `src/raitap/transparency/contracts.py`

```python
class ExplanationPayloadKind(StrEnum):
    ATTRIBUTIONS = "attributions"
    STRUCTURED = "structured"

class ExplanationScope(StrEnum):
    LOCAL = "local"
    COHORT = "cohort"
    GLOBAL = "global"

class ScopeDefinitionStep(StrEnum):
    EXPLAINER_OUTPUT = "explainer_output"
    VISUALISER_SUMMARY = "visualiser_summary"

class ExplanationOutputSpace(StrEnum):
    INPUT_FEATURES = "input_features"
    INTERPRETABLE_FEATURES = "interpretable_features"
    LAYER_ACTIVATION = "layer_activation"
    IMAGE_SPATIAL_MAP = "image_spatial_map"
    TOKEN_SEQUENCE = "token_sequence"

class InputKind(StrEnum):
    IMAGE = "image"
    TABULAR = "tabular"
    TEXT = "text"
    TIME_SERIES = "time_series"

class TensorLayout(StrEnum):
    BATCH_CHANNEL_HEIGHT_WIDTH = "NCHW"
    BATCH_FEATURE = "(B,F)"
    BATCH_TIME_CHANNEL = "(B,T,C)"
    TOKEN_SEQUENCE = "TOKENS"

class MethodFamily(StrEnum):
    GRADIENT = "gradient"
    PERTURBATION = "perturbation"
    SHAPLEY = "shapley"
    CAM = "cam"
    MODEL_AGNOSTIC = "model_agnostic"
    TREE = "tree"
    SURROGATE = "surrogate"
```

These enums are the vocabulary the transparency layer uses to describe what an explanation means and what a visualiser can consume.

## Transparency dataclasses

Source: `src/raitap/transparency/contracts.py`

```python
@dataclass(frozen=True)
class ExplanationTarget:
    target: int | str | Sequence[int] | None = None
    label: str | None = None

@dataclass(frozen=True)
class SampleSelection:
    sample_ids: list[str] | None
    sample_display_names: list[str] | None

@dataclass(frozen=True init=False)
class InputSpec:
    kind: InputKind | None
    shape: tuple[int, ...] | None
    layout: TensorLayout | None
    feature_names: list[str] | None = None
    metadata: Mapping[str, Any] | None = None

@dataclass(frozen=True init=False)
class OutputSpaceSpec:
    space: ExplanationOutputSpace
    shape: tuple[int, ...] | None
    layout: TensorLayout | None
    layer_path: str | None = None
    feature_names: list[str] | None = None
    requires_interpolation: bool = False

@dataclass(frozen=True)
class VisualSummarySpec:
    summary_type: str
    aggregation: str | None = None
    description: str | None = None

@dataclass(frozen=True)
class ExplanationSemantics:
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
    scope: ExplanationScope
    scope_definition_step: ScopeDefinitionStep
    payload_kind: ExplanationPayloadKind
    method_families: frozenset[MethodFamily]
    candidate_output_spaces: frozenset[ExplanationOutputSpace]
```

`InputSpec` and `OutputSpaceSpec` are the most operationally important types. They are what let RAITAP distinguish, for example, a token sequence from an image tensor before a visualiser runs.

## Output dataclasses

```python
@dataclass(frozen=True)
class PredictionSummary:
    sample_index: int
    predicted_class: int
    confidence: float
    sample_id: str | None = None
    target_class: int | None = None
    correct: bool | None = None

@dataclass(frozen=True)
class RunOutputs:
    explanations: list[ExplanationResult]
    visualisations: list[VisualisationResult]
    metrics: MetricsEvaluation | None
    forward_output: torch.Tensor
    sample_ids: list[str] | None = None
    targets: torch.Tensor | None = None
    prediction_summaries: tuple[PredictionSummary, ...] = ()

@dataclass
class MetricResult:
    metrics: dict[str, float]
    artifacts: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ConfiguredVisualiser:
    visualiser: BaseVisualiser
    call_kwargs: dict[str, Any] = field(default_factory=dict)
```

These types are the handoff objects between major pipeline stages.

## Reporting structures

```python
@dataclass(frozen=True slots=True)
class ReportGroup:
    heading: str
    images: tuple[Path, ...] = ()
    table_rows: tuple[tuple[str, str], ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True slots=True)
class ReportSection:
    title: str
    groups: tuple[ReportGroup, ...]
    metadata: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class ReportManifest:
    kind: str
    sections: tuple[ReportSection, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    filename: str = "report.pdf"
```

These types matter because the report builder and reporter are intentionally decoupled. `build_report()` assembles the structure; `PDFReporter` only renders it.

## Protocols

Two public protocols shape extension points:

- `ExplainerAdapter` in `src/raitap/transparency/contracts.py`
- `Trackable` in `src/raitap/tracking/base_tracker.py`

They matter because factories instantiate arbitrary Hydra targets and then validate behavior structurally instead of relying only on inheritance.
