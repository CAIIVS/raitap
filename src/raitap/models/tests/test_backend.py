"""Tests for ModelBackend.task_kind (issue #146 groundwork)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from raitap.models.torch_backend import TorchBackend
from raitap.types import Capability, TaskKind


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


def test_torch_backend_autograd_module_returns_live_module() -> None:
    import torch.nn as nn

    from raitap.models.torch_backend import TorchBackend

    module = nn.Linear(2, 2)
    backend = TorchBackend(module)
    assert backend.autograd_module() is module


def test_predict_callable_runs_forward() -> None:
    import torch
    import torch.nn as nn

    from raitap.models.torch_backend import TorchBackend

    backend = TorchBackend(nn.Identity())
    fn = backend.predict_callable()
    out = fn(torch.ones(1, 3))
    assert torch.equal(out, torch.ones(1, 3))


def test_register_backend_sets_class_constant() -> None:
    from raitap.models.base_backend import ModelBackend
    from raitap.models.registration import register

    @register(provides=frozenset({Capability.AUTOGRAD}))
    class _B(ModelBackend):
        @property
        def hardware_label(self) -> str:
            return "test"

        def __call__(self, inputs: object) -> object:
            return inputs

    assert _B.provides == frozenset({Capability.AUTOGRAD})
