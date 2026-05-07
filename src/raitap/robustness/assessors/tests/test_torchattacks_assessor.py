from __future__ import annotations

import pytest
import torch

from raitap.robustness.assessors import TorchattacksAssessor
from raitap.robustness.exceptions import AssessorBackendIncompatibilityError

torchattacks = pytest.importorskip("torchattacks")


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


def test_check_backend_compat_rejects_non_autograd_backend() -> None:
    assessor = TorchattacksAssessor(algorithm="FGSM", eps=0.03)
    with pytest.raises(AssessorBackendIncompatibilityError):
        assessor.check_backend_compat(_OnnxLikeBackend())


def test_check_backend_compat_accepts_autograd_backend() -> None:
    assessor = TorchattacksAssessor(algorithm="FGSM", eps=0.03)
    assessor.check_backend_compat(_AutogradBackend())  # no raise


def test_generate_adversarial_runs_with_fgsm() -> None:
    torch.manual_seed(0)
    model = _TinyClassifier()
    inputs = torch.rand(4, 3, 4, 4)
    targets = torch.tensor([0, 1, 2, 0])

    assessor = TorchattacksAssessor(algorithm="FGSM", eps=0.05)
    perturbed = assessor.generate_adversarial(model, inputs, targets)
    assert perturbed.shape == inputs.shape
    delta = (perturbed - inputs).abs()
    # FGSM step bounded by eps in L_inf.
    assert float(delta.max().item()) <= 0.05 + 1e-5
