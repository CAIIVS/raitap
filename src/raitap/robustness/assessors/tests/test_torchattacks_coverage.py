"""torchattacks full coverage + dispatchability + JSMA guard (#266)."""

from __future__ import annotations

import pytest
import torch

from raitap.robustness.assessors.torchattacks_assessor import TorchattacksAssessor
from raitap.robustness.contracts import Objective, PerturbationNorm, ThreatModel

torchattacks = pytest.importorskip("torchattacks")


def test_registry_size_is_36() -> None:
    assert len(TorchattacksAssessor.algorithm_registry) == 36


def test_excluded_attacks_absent() -> None:
    reg = TorchattacksAssessor.algorithm_registry
    for name in ("LGV", "MultiAttack", "VANILA"):
        assert name not in reg


def test_every_attack_constructs_and_has_valid_hints() -> None:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 10)).eval()
    for name, hints in TorchattacksAssessor.algorithm_registry.items():
        assert isinstance(hints.threat_model, ThreatModel)
        assert isinstance(hints.objective, Objective)
        assert hints.norm is None or isinstance(hints.norm, PerturbationNorm)
        assert isinstance(hints.stochastic, bool)
        getattr(torchattacks, name)(model)  # ctor must accept (model) + defaults


def _model(n: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, n)).eval()


def test_jsma_raises_on_non_10_class() -> None:
    a = TorchattacksAssessor("JSMA")
    with pytest.raises(ValueError, match="10"):
        a.generate_adversarial(_model(7), torch.rand(2, 3, 4, 4), torch.zeros(2, dtype=torch.long))


def test_jsma_runs_on_10_class() -> None:
    a = TorchattacksAssessor("JSMA")
    out = a.generate_adversarial(
        _model(10), torch.rand(2, 3, 4, 4), torch.zeros(2, dtype=torch.long)
    )
    assert out.shape == (2, 3, 4, 4)


def test_autoattack_is_self_seeded() -> None:
    assert TorchattacksAssessor.algorithm_registry["AutoAttack"].seeding == "self_seeded"


def test_pgd_is_global_rng() -> None:
    assert TorchattacksAssessor.algorithm_registry["PGD"].seeding == "global_rng"


def test_every_torchattacks_entry_declares_seeding() -> None:
    for spec in TorchattacksAssessor.algorithm_registry.values():
        assert spec.seeding in {"deterministic", "global_rng", "self_seeded"}
