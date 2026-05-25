from __future__ import annotations

import pytest
import torch

from raitap.robustness.assessors import FoolboxAssessor, TorchattacksAssessor
from raitap.robustness.contracts import (
    AssessmentKind,
    Objective,
    PerturbationNorm,
    ThreatModel,
)
from raitap.robustness.semantics import (
    assessor_semantics,
    hints_for_assessor,
)


def test_torchattacks_registry_covers_pgd() -> None:
    hints = TorchattacksAssessor.algorithm_registry["PGD"]
    assert hints.assessment_kind == AssessmentKind.EMPIRICAL_ATTACK
    assert hints.norm == PerturbationNorm.LINF
    assert "iterative" in hints.families


def test_foolbox_registry_distinguishes_l2_from_linf() -> None:
    assert FoolboxAssessor.algorithm_registry["LinfPGD"].norm == PerturbationNorm.LINF
    assert FoolboxAssessor.algorithm_registry["L2PGD"].norm == PerturbationNorm.L2


def test_hints_for_assessor_routes_to_torchattacks() -> None:
    assessor = TorchattacksAssessor(algorithm="FGSM")
    hints = hints_for_assessor(assessor)
    assert hints.threat_model == ThreatModel.WHITE_BOX


def test_hints_for_assessor_routes_to_foolbox() -> None:
    assessor = FoolboxAssessor(algorithm="LinfPGD")
    hints = hints_for_assessor(assessor)
    assert hints.assessment_kind == AssessmentKind.EMPIRICAL_ATTACK


def test_assessor_semantics_reads_budget_from_constructor_kwargs() -> None:
    # Torchattacks puts eps / alpha / steps under YAML ``constructor:`` because those
    # are the attack class's __init__ kwargs. The semantics resolver must look there.
    assessor = TorchattacksAssessor(algorithm="PGD", eps=0.04, alpha=0.005, steps=12)
    inputs = torch.randn(2, 3, 4, 4)
    targets = torch.tensor([0, 1])
    semantics = assessor_semantics(
        assessor,
        call_kwargs={},
        raitap_kwargs={},
        inputs=inputs,
        targets=targets,
    )
    assert semantics.budget.epsilon == 0.04
    assert semantics.budget.step_size == 0.005
    assert semantics.budget.steps == 12


def test_assessor_semantics_extracts_targeted_objective_from_call_kwargs() -> None:
    """Targeted-mode detection always reads call_kwargs regardless of framework."""
    assessor = TorchattacksAssessor(algorithm="PGD", eps=0.05, alpha=0.01, steps=7)
    inputs = torch.randn(2, 3, 4, 4)
    targets = torch.tensor([0, 1])
    semantics = assessor_semantics(
        assessor,
        call_kwargs={"target_labels": [3, 4]},
        raitap_kwargs={"input_metadata": {"kind": "image", "layout": "NCHW"}},
        inputs=inputs,
        targets=targets,
    )
    assert semantics.assessment_kind == AssessmentKind.EMPIRICAL_ATTACK
    assert semantics.objective == Objective.TARGETED
    assert semantics.target_classes == (3, 4)
    # Budget reflects the constructor (where torchattacks actually reads from).
    assert semantics.budget.epsilon == 0.05
    assert semantics.budget.step_size == 0.01
    assert semantics.budget.steps == 7


def test_assessor_semantics_foolbox_reads_budget_from_call_kwargs() -> None:
    """Foolbox consumes the budget at call time, so semantics must read call_kwargs."""
    assessor = FoolboxAssessor(algorithm="LinfPGD")
    inputs = torch.randn(2, 3, 4, 4)
    targets = torch.tensor([0, 1])
    semantics = assessor_semantics(
        assessor,
        call_kwargs={"eps": 0.07, "steps": 25},
        raitap_kwargs={},
        inputs=inputs,
        targets=targets,
    )
    assert semantics.budget.epsilon == 0.07
    assert semantics.budget.steps == 25


def test_assessor_semantics_warns_on_misplaced_budget_keys() -> None:
    """Putting budget under the wrong YAML block must surface a clear warning."""
    assessor = TorchattacksAssessor(algorithm="PGD")  # reads init_kwargs
    inputs = torch.randn(1, 3, 4, 4)
    targets = torch.tensor([0])
    with pytest.warns(UserWarning, match="ignored by the adapter"):
        semantics = assessor_semantics(
            assessor,
            call_kwargs={"eps": 0.05},  # wrong source for torchattacks
            raitap_kwargs={},
            inputs=inputs,
            targets=targets,
        )
    # Misplaced kwargs in call_kwargs are not consumed by the adapter, so the
    # resulting budget reflects init_kwargs (empty) plus registry default.
    assert semantics.budget.epsilon is None
