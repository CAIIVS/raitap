"""
Models module — loads pretrained or custom models and selects a backend
(native PyTorch / ONNX).

Public Python surface
---------------------
``ModelConfig``
    Dataclass section of ``AppConfig`` describing how to locate / construct
    the model under test.
``Model``
    The loaded model object passed to assessors and explainers.
"""

from raitap.configs.schema import ModelConfig

from .model import Model

__all__ = [
    "Model",
    "ModelConfig",
]
