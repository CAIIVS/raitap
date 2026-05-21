# src/raitap/robustness/assessors/tests/test_torchattacks_parity.py
"""Tier-1 parity: raitap's TorchattacksAssessor produces the same adversarial
inputs as a direct torchattacks PGD call for the same model/inputs/config/seed.
Catches dropped kwargs, wrong wiring, or stray mutations in the data path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from raitap.robustness.assessors import TorchattacksAssessor
from raitap.testing import make_tiny_classifier

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.e2e, pytest.mark.parity, pytest.mark.robustness]

_RTOL = 1e-5
_ATOL = 1e-6


@pytest.fixture
def _ta() -> Any:
    return pytest.importorskip("torchattacks")


def test_pgd_adversarials_match_direct_call(_ta: Any, seeded: Callable[..., None]) -> None:
    seeded()
    model = make_tiny_classifier(seed=0)
    x = torch.rand(2, 3, 8, 8)  # PGD expects inputs in [0, 1]
    y = torch.tensor([0, 1])

    # Direct torchattacks call — ground truth.
    atk = _ta.PGD(model, eps=8 / 255, alpha=2 / 255, steps=5, random_start=False)
    raw_adv = atk(x, y)

    # raitap path: same PGD config forwarded via TorchattacksAssessor.
    assessor = TorchattacksAssessor(
        algorithm="PGD",
        eps=8 / 255,
        alpha=2 / 255,
        steps=5,
        random_start=False,
    )
    raitap_adv = assessor.generate_adversarial(model, x, y)

    assert torch.allclose(
        raw_adv.detach().cpu(),
        raitap_adv.detach().cpu(),
        rtol=_RTOL,
        atol=_ATOL,
    )
