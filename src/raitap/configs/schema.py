from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TransparencyFramework(StrEnum):
    captum = "captum"
    shap = "shap"


@dataclass
class ModelConfig:
    name: str = "default_model"
    pretrained: bool = True


@dataclass
class DataConfig:
    name: str = "isic2018"
    description: str | None = None
    directory: str | None = None


@dataclass
class TransparencyConfig:
    framework: TransparencyFramework = TransparencyFramework.captum
    algorithm: str = "integrated_gradients"
    output_dir: str = "outputs/transparency"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transparency: TransparencyConfig = field(default_factory=TransparencyConfig)
    experiment_name: str = "mvp"
