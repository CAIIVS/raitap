from __future__ import annotations

import torch

from raitap.robustness.assessors import FoolboxAssessor, TorchattacksAssessor
from raitap.robustness.contracts import (
    MethodKind,
    Objective,
    PerturbationNorm,
    ThreatModel,
)
from raitap.robustness.semantics import (
    FOOLBOX_REGISTRY,
    TORCHATTACKS_REGISTRY,
    assessor_semantics,
    hints_for_assessor,
)


def test_torchattacks_registry_covers_pgd() -> None:
    hints = TORCHATTACKS_REGISTRY["PGD"]
    assert hints.method_kind == MethodKind.EMPIRICAL_ATTACK
    assert hints.norm == PerturbationNorm.LINF
    assert "iterative" in hints.families


def test_foolbox_registry_distinguishes_l2_from_linf() -> None:
    assert FOOLBOX_REGISTRY["LinfPGD"].norm == PerturbationNorm.LINF
    assert FOOLBOX_REGISTRY["L2PGD"].norm == PerturbationNorm.L2


def test_hints_for_assessor_routes_to_torchattacks() -> None:
    assessor = TorchattacksAssessor(algorithm="FGSM")
    hints = hints_for_assessor(assessor)
    assert hints.threat_model == ThreatModel.WHITE_BOX


def test_hints_for_assessor_routes_to_foolbox() -> None:
    assessor = FoolboxAssessor(algorithm="LinfPGD")
    hints = hints_for_assessor(assessor)
    assert hints.method_kind == MethodKind.EMPIRICAL_ATTACK


def test_assessor_semantics_extracts_budget_and_targeted_objective() -> None:
    assessor = TorchattacksAssessor(algorithm="PGD")
    inputs = torch.randn(2, 3, 4, 4)
    targets = torch.tensor([0, 1])
    semantics = assessor_semantics(
        assessor,
        call_kwargs={"eps": 0.05, "alpha": 0.01, "steps": 7, "target_labels": [3, 4]},
        raitap_kwargs={"input_metadata": {"kind": "image", "layout": "NCHW"}},
        inputs=inputs,
        targets=targets,
    )
    assert semantics.method_kind == MethodKind.EMPIRICAL_ATTACK
    assert semantics.objective == Objective.TARGETED
    assert semantics.budget.epsilon == 0.05
    assert semantics.budget.step_size == 0.01
    assert semantics.budget.steps == 7
    assert semantics.target_classes == (3, 4)
