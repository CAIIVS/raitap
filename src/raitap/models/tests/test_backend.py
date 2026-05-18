"""Tests for ModelBackend.task_kind (issue #146 groundwork)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from raitap.models.backend import TorchBackend
from raitap.types import TaskKind


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_torch_backend_defaults_to_classification() -> None:
    backend = TorchBackend(_Linear())
    assert backend.task_kind is TaskKind.classification


def test_torch_backend_detects_torchvision_detection_model() -> None:
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    except ImportError:
        pytest.skip("torchvision detection models unavailable")

    model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=2)
    backend = TorchBackend(model)
    assert backend.task_kind is TaskKind.detection


def test_torch_backend_task_kind_can_be_overridden_in_constructor() -> None:
    backend = TorchBackend(_Linear(), task_kind=TaskKind.regression)
    assert backend.task_kind is TaskKind.regression
