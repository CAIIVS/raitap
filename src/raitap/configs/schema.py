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


@dataclass
class TransparencyConfig:
    # Hydra _target_: points to a BaseExplainer subclass.
    # Overridden by the transparency config-group YAML (transparency=captum / shap).
    _target_: str = "CaptumExplainer"
    algorithm: str = "IntegratedGradients"
    # Each item is a dict/DictConfig with a ``_target_`` key pointing to a
    # BaseVisualiser subclass.  Additional keys are forwarded to the constructor.
    visualisers: list[Any] = field(default_factory=lambda: [{"_target_": "CaptumImageVisualiser"}])


@dataclass
class MetricsConfig:
    _target_: str = "ClassificationMetrics"
    task: str = "multiclass"
    num_classes: int | None = None


@dataclass
class TrackingConfig:
    output_forwarding_url: str | None = None
    log_model: bool = False
    registry_enabled: bool = False
    registered_model_name: str | None = None
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
