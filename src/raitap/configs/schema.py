from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    # Path to a local .pth file, or a built-in name (e.g. "resnet50")
    source: str | None = None


@dataclass
class DataConfig:
    name: str = "isic2018"
    description: str | None = None
    # Path to a local dir, or a named sample set (e.g. "imagenet_samples")
    source: str | None = None
    # Optional path to a labels file (currently CSV/TSV/Parquet).
    labels_source: str | None = None
    # Optional sample-id column for filename alignment (e.g. "image").
    labels_id_column: str | None = None
    # Optional class-label column; when omitted, one-hot numeric columns are used via argmax.
    labels_column: str | None = None
    # Optional parsing strategy for labels: "index", "one_hot", or "argmax".
    labels_encoding: str | None = None


@dataclass
class TransparencyConfig:
    # Hydra _target_: points to a BaseExplainer subclass.
    # Overridden by the transparency config-group YAML (transparency=captum / shap).
    _target_: str = "CaptumExplainer"
    algorithm: str = "IntegratedGradients"
    # Constructor kwargs for the explainer / underlying library method (e.g. Captum
    # ``IntegratedGradients(model, **kwargs)``, SHAP ``GradientExplainer(model, data, **kwargs)``).
    constructor: dict[str, Any] = field(default_factory=dict)
    # Per-call kwargs for ``compute_attributions`` (Captum ``.attribute()``,
    # SHAP ``.shap_values()``).
    call: dict[str, Any] = field(default_factory=dict)
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
    open_when_done: bool = False


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transparency: dict[str, Any] = field(default_factory=lambda: {"default": TransparencyConfig()})
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    experiment_name: str = "mvp"
    # Fallback output directory used when running outside of a Hydra session.
    fallback_output_dir: str = "."
