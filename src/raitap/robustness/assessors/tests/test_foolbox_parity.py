"""Parity test: FoolboxAssessor vs direct foolbox LinfPGD call."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.e2e, pytest.mark.parity, pytest.mark.robustness]


@pytest.fixture
def _fb() -> object:
    return pytest.importorskip("foolbox")


def test_pgd_advs_match_direct_call(seeded: object, _fb: object) -> None:
    import foolbox

    from raitap.robustness.assessors import FoolboxAssessor
    from raitap.testing import make_tiny_classifier

    seeded()  # type: ignore[operator]
    model = make_tiny_classifier(seed=0)
    x = torch.rand(2, 3, 8, 8)
    y = torch.tensor([0, 1])

    # Direct foolbox call — second return value is the clipped adversarial tensor.
    # random_start=False makes LinfPGD deterministic (default is True).
    fmodel = foolbox.PyTorchModel(model, bounds=(0, 1))  # pyright: ignore[reportPrivateImportUsage]
    attack = foolbox.attacks.LinfPGD(steps=5, random_start=False)
    _, raw_adv, _ = attack(fmodel, x, y, epsilons=8 / 255)

    # raitap path — same attack, same config.
    assessor = FoolboxAssessor(algorithm="LinfPGD", bounds=(0, 1), steps=5, random_start=False)
    raitap_adv = assessor.generate_adversarial(model, x, y, epsilons=8 / 255)

    assert torch.allclose(
        raw_adv.detach().cpu(),
        raitap_adv.detach().cpu(),
        rtol=1e-5,
        atol=1e-6,
    )
