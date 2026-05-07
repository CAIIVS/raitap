from __future__ import annotations

import pytest
import torch

from raitap.robustness.assessors import FoolboxAssessor
from raitap.robustness.exceptions import AssessorBackendIncompatibilityError

foolbox = pytest.importorskip("foolbox")


class _AutogradBackend:
    supports_torch_autograd = True


class _OnnxLikeBackend:
    supports_torch_autograd = False


class _TinyClassifier(torch.nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(3 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x.flatten(1))


def test_foolbox_rejects_non_autograd_backend():
    assessor = FoolboxAssessor(algorithm="LinfPGD")
    with pytest.raises(AssessorBackendIncompatibilityError):
        assessor.check_backend_compat(_OnnxLikeBackend())


def test_foolbox_rejects_list_epsilons():
    model = _TinyClassifier()
    inputs = torch.rand(2, 3, 4, 4)
    targets = torch.tensor([0, 1])
    assessor = FoolboxAssessor(algorithm="LinfFastGradientAttack")
    with pytest.raises(TypeError, match="multi-epsilon"):
        assessor.generate_adversarial(model, inputs, targets, epsilons=[0.01, 0.03])


def test_foolbox_requires_epsilon():
    model = _TinyClassifier()
    inputs = torch.rand(2, 3, 4, 4)
    targets = torch.tensor([0, 1])
    assessor = FoolboxAssessor(algorithm="LinfFastGradientAttack")
    with pytest.raises(ValueError, match="requires `call.eps`"):
        assessor.generate_adversarial(model, inputs, targets)
