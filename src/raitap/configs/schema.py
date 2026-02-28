from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from raitap.transparency.methods_registry import get_framework_names

# Dynamically generated from methods_registry - add frameworks there
TransparencyFramework = StrEnum(
    "TransparencyFramework",
    {name: name for name in get_framework_names()},
)


@dataclass
class ModelConfig:
    """
    Optional configuration for convenience model loading.
    For custom models, ignore this config and pass your model directly
    to the transparency methods.
    """

    name: str | None = None  # e.g., "resnet50", or None for custom models
    pretrained: bool = True


@dataclass
class DataConfig:
    name: str = "isic2018"
    description: str | None = None
    directory: str | None = None


@dataclass
class TransparencyConfig:
    # Uses TransparencyFramework enum (dynamically generated from registry)
    framework: str = "captum"
    algorithm: str = "IntegratedGradients"
    # List of visualiser names valid for the chosen framework.
    # Captum supports: "image", "time_series", "text"
    # SHAP supports:   "bar", "beeswarm", "waterfall", "force", "image"
    # SHAP "image" is only compatible with GradientExplainer / DeepExplainer.
    visualisers: list[str] = field(default_factory=lambda: ["image"])
    output_dir: str = "outputs/transparency"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transparency: TransparencyConfig = field(default_factory=TransparencyConfig)
    experiment_name: str = "mvp"
