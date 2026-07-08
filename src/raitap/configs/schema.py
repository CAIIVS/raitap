from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omegaconf import MISSING

from raitap.data.types import IdStrategy, LabelEncoding
from raitap.types import Hardware, TaskKind

if TYPE_CHECKING:
    from raitap.metrics.classification_metrics import Average as ClassificationAverage
    from raitap.metrics.detection_metrics import (
        Average as DetectionAverage,
    )
    from raitap.metrics.detection_metrics import (
        Backend,
        BoxFormat,
        IoUType,
    )
else:
    # Runtime aliases so omegaconf's ``get_type_hints()`` can resolve the
    # string-form annotations on the dataclasses below without importing the
    # metrics package (which would create a circular import: metrics modules
    # depend on this schema for their typed configs). omegaconf does not
    # natively support ``Literal`` types, so the runtime fallback widens to
    # ``str``; the canonical narrow ``Literal`` definitions live in the
    # respective metrics modules and are enforced by adapter ``__init__``
    # validation.
    ClassificationAverage = str
    DetectionAverage = str
    Backend = str
    BoxFormat = str
    # ``IoUType`` real type is ``str | tuple[str, ...]`` — omegaconf rejects
    # unions of primitives and containers ("Unions of containers are not
    # supported"), so the runtime alias has to stay ``Any``. The narrow
    # ``Literal`` definition lives in ``raitap.metrics.detection_metrics``.
    IoUType = Any


@dataclass
class ModelConfig:
    # Path to a local .pt / .pth / .onnx file, or a built-in name (e.g. "resnet50").
    source: str | None = None
    # Architecture spec used when ``source`` points at a state-dict (``.pt`` /
    # ``.pth`` containing a ``dict``). Must be the name of a torchvision model
    # builder callable that accepts ``weights=`` and ``num_classes=`` kwargs
    # (e.g. ``"resnet18"``, ``"vit_b_16"``). Other ``torchvision.models``
    # attributes (constants, helpers) are not valid here.
    arch: str | None = None
    # Number of output classes used to instantiate the architecture before
    # ``load_state_dict``. Required together with ``arch`` for state-dict loading.
    num_classes: int | None = None
    # Whether to construct the architecture with ImageNet pretrained weights
    # before loading the state-dict. Usually ``False`` since weights come from
    # the state-dict itself.
    pretrained: bool = False
    # Optional explicit id->name table for detection class labels. Index-aligned
    # with the model's label ids (id 0 first). When set, overrides any names
    # captured from torchvision ``weights.meta["categories"]``. Leave unset to
    # use the model's bundled category names (pretrained torchvision detectors)
    # or fall back to numeric ids.
    class_names: list[str] | None = None
    # Optional explicit task family. Leave unset to auto-infer from the model
    # architecture (torchvision detectors are detected automatically). Set to
    # the TaskKind value (e.g. "detection") for custom models the inference
    # can't recognise.
    task_kind: TaskKind | None = None
    # HuggingFace tokenizer id/path for text models. Set => text modality.
    tokenizer: str | None = None


@dataclass
class LabelsConfig:
    use: str = MISSING


@dataclass
class InputsConfig:
    use: str = MISSING


@dataclass
class TextCsvInputsConfig(InputsConfig):
    use: str = "text_csv"
    text_column: str = MISSING


@dataclass
class TextJsonlInputsConfig(InputsConfig):
    use: str = "text_jsonl"
    text_field: str = "text"


@dataclass
class TextDirInputsConfig(InputsConfig):
    use: str = "text_dir"


@dataclass
class TabularLabelsConfig(LabelsConfig):
    use: str = "tabular"
    source: str = MISSING
    id_column: str | None = None
    column: str | None = None
    encoding: LabelEncoding | None = None
    id_strategy: IdStrategy = IdStrategy.auto


@dataclass
class DirectoryLabelsConfig(LabelsConfig):
    use: str = "directory"


@dataclass
class CocoLabelsConfig(LabelsConfig):
    use: str = "coco"
    source: str = MISSING
    id_strategy: IdStrategy = IdStrategy.auto


@dataclass
class YoloLabelsConfig(LabelsConfig):
    use: str = "yolo"
    source: str = MISSING
    id_strategy: IdStrategy = IdStrategy.auto


@dataclass
class VocLabelsConfig(LabelsConfig):
    use: str = "voc"
    source: str = MISSING
    id_strategy: IdStrategy = IdStrategy.auto
    class_names: list[str] | None = None


@dataclass
class DetectionJsonLabelsConfig(LabelsConfig):
    use: str = "detection_json"
    source: str = MISSING
    id_strategy: IdStrategy = IdStrategy.auto


@dataclass
class DataConfig:
    name: str = "isic2018"
    description: str | None = None
    # Path to a local dir, or a named sample set (e.g. "imagenet_samples")
    source: str | None = None
    # Optional model-forward batch size for predictions/metrics. None uses the pipeline default.
    forward_batch_size: int | None = None
    # Data-side preprocessing applied per-image in the loader, before the
    # batch is stacked. Typical contents: Resize, CenterCrop. Independent of
    # ``model_input_transformation``. Accepts:
    #   - ``None`` (default): no data preprocessing.
    #   - ``Preprocessing.model_bundled`` (or the string ``"model-bundled"``):
    #     pull Resize + CenterCrop from the resolved torchvision arch's bundled
    #     preset (``Weights.transforms()``).
    #   - path to a ``.py`` file with an ``@raitap_preprocessing_factory``.
    # File loading is gated by ``acknowledge_preprocessing_exec`` (Python API)
    # or ``--allow-preprocessing-exec`` / ``-yp`` (CLI).
    preprocessing: str | None = None
    # Transformation applied at the model boundary, on every forward pass.
    # Stays inside autograd so attribution and adversarial budgets see the
    # user-facing input space. Typical contents: Normalize. Independent of
    # ``preprocessing``. Same accepted values as ``preprocessing`` but the
    # file factory must be decorated with
    # ``@raitap_model_input_transformation_factory``.
    # When both knobs are ``None`` and inputs are images, a loud warning fires
    # at startup; silence with ``acknowledge_preprocessing_off`` /
    # ``--acknowledge-preprocessing-off``.
    model_input_transformation: str | None = None
    # Optional input-modality metadata (``kind``, ``feature_names``, ``layout``, ...).
    # Forwarded to ``infer_input_spec`` so semantics and visualisers see the correct
    # modality for non-image data such as ACAS Xu's 5-feature tabular vector.
    input_metadata: dict[str, Any] | None = None
    labels: LabelsConfig | None = None
    inputs: InputsConfig | None = None


@dataclass
class EvaluationConfig:
    # Hydra use: selects an evaluator (e.g. "quantus" -> raitap.transparency.QuantusEvaluator).
    use: str = MISSING
    # Names of the Quantus metrics to compute (e.g. "sparseness", "faithfulness_correlation").
    metrics: list[str] = field(default_factory=list)
    # Constructor kwargs forwarded to the underlying Quantus metric classes.
    constructor: dict[str, Any] = field(default_factory=dict)
    # Per-call kwargs forwarded verbatim to the metric's ``__call__``.
    call: dict[str, Any] = field(default_factory=dict)
    # RAITAP-owned runtime options such as batch_size and progress bars. Not
    # forwarded to the underlying Quantus library.
    raitap: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransparencyConfig:
    # Hydra use: selects an ExplainerAdapter registry key
    # (e.g. "captum" -> AttributionOnlyExplainer or "shap" -> FullExplainer subclass)
    # Overridden by the transparency config-group YAML (transparency=captum / shap).
    # MISSING by default so omission fails validation loudly rather than
    # silently selecting a library.
    use: str = MISSING
    algorithm: str = MISSING
    # Constructor kwargs for the explainer / underlying library method (e.g. Captum
    # ``IntegratedGradients(model, **kwargs)``, SHAP ``GradientExplainer(model, data, **kwargs)``).
    constructor: dict[str, Any] = field(default_factory=dict)
    # Per-call kwargs forwarded verbatim to the underlying library
    # (Captum ``.attribute()``, SHAP ``.shap_values()``, etc.).
    # Any value that is a dict with a ``source`` key is treated as a data-source reference
    # and loaded as a tensor at runtime.  Example for SHAP background data::
    #
    #   call:
    #     background_data:
    #       source: "imagenet_samples"
    #       n_samples: 50
    call: dict[str, Any] = field(default_factory=dict)
    # RAITAP-owned runtime options such as batch_size, progress bars, and
    # sample-name metadata. These keys are not forwarded to the explainability library.
    raitap: dict[str, Any] = field(default_factory=dict)
    # Each entry needs at least ``use``; ``constructor`` / ``call`` are optional
    # (same split as explainer). Default is minimal: Captum explainer + image visualiser.
    visualisers: list[Any] = field(default_factory=lambda: [{"use": "captum_image"}])
    # Optional Quantus-backed explanation-quality evaluation block. Left unset
    # (``None``) by default; set ``evaluation.use`` to enable it.
    evaluation: EvaluationConfig | None = None


@dataclass
class RobustnessConfig:
    # Hydra use: selects a BaseAssessor subclass registry key
    # (e.g. "torchattacks" -> EmpiricalAttackAssessor or "marabou" -> FormalVerificationAssessor).
    # Overridden by the robustness config-group YAML
    # (robustness=torchattacks / foolbox / marabou).
    use: str = MISSING
    algorithm: str = MISSING
    # Constructor kwargs forwarded to the assessor's ``__init__``. For torchattacks
    # adapters this is where attack hyperparameters live (eps, alpha, steps), since
    # torchattacks consumes them at attack-instance construction.
    constructor: dict[str, Any] = field(default_factory=dict)
    # Per-call kwargs forwarded verbatim to the assessor at ``assess()`` time. For
    # foolbox adapters this is where the runtime budget (eps / epsilons) lives.
    # Any value that is a dict with a ``source`` key is treated as a data-source
    # reference and loaded as a tensor at runtime.
    call: dict[str, Any] = field(default_factory=dict)
    # RAITAP-owned runtime options such as batch_size, progress bars, and
    # sample-name metadata. Not forwarded to the underlying library.
    raitap: dict[str, Any] = field(default_factory=dict)
    # Each entry needs at least ``use``; ``constructor`` / ``call`` are optional.
    # Default is the empirical image-pair visualiser.
    visualisers: list[Any] = field(default_factory=lambda: [{"use": "image_pair"}])


@dataclass
class MetricsConfig:
    use: str = MISSING


@dataclass
class IoUConfig:
    type: IoUType = "bbox"
    thresholds: list[float] | None = None
    rec_thresholds: list[float] | None = None
    max_detection_thresholds: list[int] | None = None


@dataclass
class BinaryClassificationMetricsConfig(MetricsConfig):
    use: str = "binary_classification"
    ignore_index: int | None = None
    threshold: float = 0.5


@dataclass
class MulticlassClassificationMetricsConfig(MetricsConfig):
    use: str = "multiclass_classification"
    num_classes: int = MISSING
    average: ClassificationAverage = "macro"
    ignore_index: int | None = None


@dataclass
class MultilabelClassificationMetricsConfig(MetricsConfig):
    use: str = "multilabel_classification"
    num_labels: int = MISSING
    average: ClassificationAverage = "macro"
    ignore_index: int | None = None
    threshold: float = 0.5


@dataclass
class DetectionMetricsConfig(MetricsConfig):
    use: str = "detection"
    box_format: BoxFormat = "xyxy"
    iou: IoUConfig = field(default_factory=IoUConfig)
    class_metrics: bool = False
    extended_summary: bool = False
    average: DetectionAverage = "macro"
    backend: Backend = "faster_coco_eval"


@dataclass
class TrackingConfig:
    use: str = MISSING
    output_forwarding_url: str | None = None
    backend_store_uri: str | None = None
    default_artifact_root: str | None = None
    log_model: bool = False
    open_when_done: bool = True


@dataclass
class ReportingConfig:
    """Configuration for report generation."""

    # ``None`` disables reporting entirely (used by ``reporting/disabled.yaml``).
    use: str | None = MISSING
    filename: str = "report"
    sample_selection: list[int | str] | None = None
    include_config: bool = True
    include_metadata: bool = True
    multirun_report: bool = True
    show_original_per_explainer: bool = False
    show_redundant_robustness_panels: bool = False
    call: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transparency: dict[str, TransparencyConfig] = field(default_factory=dict)
    robustness: dict[str, RobustnessConfig] = field(default_factory=dict)
    metrics: MetricsConfig | None = None
    tracking: TrackingConfig | None = None
    reporting: ReportingConfig | None = None  # Optional, null by default
    hardware: Hardware = Hardware.gpu
    experiment_name: str = "Experiment"
    # Optional RNG seed. When set, RAITAP pins the process-global torch / numpy /
    # random RNGs at run start, making ``global_rng`` stochastic methods
    # reproducible. Recorded in the run dir via the resolved-config dump.
    # ``self_seeded`` methods still need their own seed param. (issue #339)
    seed: int | None = None
