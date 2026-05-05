from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ModelConfig:
    # Path to a local .pth file, or a built-in name (e.g. "resnet50")
    source: str | None = None


@dataclass
class LabelsConfig:
    # Optional path to a labels file (currently CSV/TSV/Parquet).
    source: str | None = None
    # Optional sample-id column for filename alignment (e.g. "image").
    id_column: str | None = None
    # Optional class-label column; when omitted, one-hot numeric columns are used via argmax.
    column: str | None = None
    # Optional parsing strategy for labels: "index", "one_hot", or "argmax".
    encoding: str | None = None
    # Strategy for matching label-file ids to discovered sample files:
    #   "auto"          — pick "relative_path" if any id contains "/" or "\\"; else "stem".
    #   "relative_path" — ids are resolved as posix-style paths relative to ``data.source``
    #                     (supports nested ImageFolder layouts with colliding stems).
    #   "stem"          — legacy flat-dir behaviour: match by ``Path(id).stem`` only.
    id_strategy: Literal["auto", "relative_path", "stem"] = "auto"


@dataclass
class DataConfig:
    name: str = "isic2018"
    description: str | None = None
    # Path to a local dir, or a named sample set (e.g. "imagenet_samples")
    source: str | None = None
    # Optional model-forward batch size for predictions/metrics. None uses the pipeline default.
    forward_batch_size: int | None = None
    labels: LabelsConfig = field(default_factory=LabelsConfig)


@dataclass
class TransparencyConfig:
    # Hydra _target_: points to an ExplainerAdapter
    # (e.g. AttributionOnlyExplainer or FullExplainer subclass)
    # Overridden by the transparency config-group YAML (transparency=captum / shap).
    _target_: str = "CaptumExplainer"
    algorithm: str = "IntegratedGradients"
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
    # Each entry needs at least ``_target_``; ``constructor`` / ``call`` are optional
    # (same split as explainer). Default is minimal: Captum explainer + image visualiser.
    visualisers: list[Any] = field(default_factory=lambda: [{"_target_": "CaptumImageVisualiser"}])


@dataclass
class MetricsConfig:
    _target_: str = "ClassificationMetrics"
    task: str = "multiclass"
    num_classes: int | None = None


@dataclass
class TrackingConfig:
    _target_: str = "MLFlowTracker"
    output_forwarding_url: str | None = None
    log_model: bool = False
    open_when_done: bool = True


@dataclass
class ReportingFormattingConfig:
    """PDF layout and rasterisation tuning for embedded figures."""

    # None = derive from A4 column size / heuristic page budget.
    max_image_width_pt: int | None = None
    max_image_height_pt: int | None = None
    figures_max_pages: int | None = None
    # Pixels-per-layout-point (borb draws at ``size`` points). None ≈ 3 (~216 DPI).
    image_raster_multiplier: float | None = None
    image_raster_max_edge_px: int | None = None


@dataclass
class ReportingConfig:
    """Configuration for report generation."""

    _target_: str = "PDFReporter"
    filename: str = "report.pdf"
    include_config: bool = True
    include_metadata: bool = True
    multirun_report: bool = True
    formatting: ReportingFormattingConfig = field(default_factory=ReportingFormattingConfig)


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transparency: dict[str, Any] = field(default_factory=lambda: {"default": TransparencyConfig()})
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reporting: ReportingConfig | None = None  # Optional, null by default
    hardware: str = "gpu"
    experiment_name: str = "Experiment"
