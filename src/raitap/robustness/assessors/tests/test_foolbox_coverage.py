"""foolbox coverage + dispatchability + DatasetAttack feed invoker (#266)."""

from __future__ import annotations

import pytest
import torch

from raitap.robustness.assessors.foolbox_assessor import FoolboxAssessor
from raitap.robustness.contracts import Objective, PerturbationNorm, ThreatModel

foolbox = pytest.importorskip("foolbox")

from foolbox import attacks as A  # noqa: E402, N812

# Attacks whose ctor requires a positional/required arg (no usable defaults) get minimal
# kwargs so the bare-construct check below can instantiate them.
_CTOR_KWARGS: dict[str, dict] = {"VirtualAdversarialAttack": {"steps": 1}}


def test_registry_size() -> None:
    assert len(FoolboxAssessor.algorithm_registry) == 55


def test_excluded_attacks_absent() -> None:
    # Deferred (starting-point / lifecycle attacks):
    #   BinarizationRefinementAttack #277, SpatialAttack #278, PointwiseAttack #280
    reg = FoolboxAssessor.algorithm_registry
    for name in ("Attack", "BinarizationRefinementAttack", "SpatialAttack", "PointwiseAttack"):
        assert name not in reg


def test_no_alias_duplicates() -> None:
    # Every registered name maps to a distinct foolbox class (no alias double-registration).
    classes = [getattr(A, name) for name in FoolboxAssessor.algorithm_registry if hasattr(A, name)]
    assert len(classes) == len({id(c) for c in classes})


def test_every_default_attack_constructs_and_has_valid_hints() -> None:
    for name, hints in FoolboxAssessor.algorithm_registry.items():
        assert isinstance(hints.threat_model, ThreatModel)
        assert isinstance(hints.objective, Objective)
        assert hints.norm is None or isinstance(hints.norm, PerturbationNorm)
        assert isinstance(hints.stochastic, bool)
        if hints.invoker is None:
            # default path: bare ctor (or minimal kwargs for config-required attacks)
            getattr(A, name)(**_CTOR_KWARGS.get(name, {}))


def test_dataset_attack_registered_and_runs() -> None:
    a = FoolboxAssessor("DatasetAttack")
    assert FoolboxAssessor.algorithm_registry["DatasetAttack"].invoker is not None
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 10)).eval()
    out = a.generate_adversarial(
        model, torch.rand(2, 3, 4, 4), torch.zeros(2, dtype=torch.long), eps=0.3
    )
    assert out.shape == (2, 3, 4, 4)


def test_dataset_attack_honors_targeted_criterion() -> None:
    # The DatasetAttack invoker builds a targeted criterion (not silently dropped).
    a = FoolboxAssessor("DatasetAttack")
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 10)).eval()
    out = a.generate_adversarial(
        model,
        torch.rand(2, 3, 4, 4),
        torch.zeros(2, dtype=torch.long),
        eps=0.3,
        target_labels=torch.ones(2, dtype=torch.long),
    )
    assert out.shape == (2, 3, 4, 4)
