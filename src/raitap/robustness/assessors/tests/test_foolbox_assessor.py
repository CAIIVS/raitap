from __future__ import annotations

import pytest
import torch

from raitap.robustness.assessors import FoolboxAssessor
from raitap.testing import make_fake_backend, make_pixel_linear_classifier
from raitap.utils.errors import BackendIncompatibilityError

foolbox = pytest.importorskip("foolbox")


def test_foolbox_rejects_non_autograd_backend() -> None:
    assessor = FoolboxAssessor(algorithm="LinfPGD")
    with pytest.raises(BackendIncompatibilityError):
        assessor.check_backend_compat(make_fake_backend(provides=frozenset()))


def test_foolbox_rejects_list_epsilons() -> None:
    model = make_pixel_linear_classifier(hw=4)
    inputs = torch.rand(2, 3, 4, 4)
    targets = torch.tensor([0, 1])
    assessor = FoolboxAssessor(algorithm="LinfFastGradientAttack")
    with pytest.raises(TypeError, match="multi-epsilon"):
        assessor.generate_adversarial(model, inputs, targets, epsilons=[0.01, 0.03])


def test_foolbox_requires_epsilon() -> None:
    model = make_pixel_linear_classifier(hw=4)
    inputs = torch.rand(2, 3, 4, 4)
    targets = torch.tensor([0, 1])
    assessor = FoolboxAssessor(algorithm="LinfFastGradientAttack")
    with pytest.raises(ValueError, match=r"requires `call\.eps`"):
        assessor.generate_adversarial(model, inputs, targets)
