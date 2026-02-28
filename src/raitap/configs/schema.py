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
    Configuration for model loading.
    """

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
    framework: str = "captum"
    algorithm: str = "IntegratedGradients"
    # List of visualiser names valid for the chosen framework.
    # Captum supports: "image", "time_series", "text"
    # SHAP supports:   "bar", "beeswarm", "waterfall", "force", "image"
    # SHAP "image" is only compatible with GradientExplainer / DeepExplainer.
    visualisers: list[str] = field(default_factory=lambda: ["image"])

    def __post_init__(self) -> None:
        valid = set(TransparencyFramework)
        if self.framework not in valid:
            raise ValueError(
                f"Unknown framework {self.framework!r}. Valid options: {sorted(valid)}"
            )


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transparency: TransparencyConfig = field(default_factory=TransparencyConfig)
    experiment_name: str = "mvp"
