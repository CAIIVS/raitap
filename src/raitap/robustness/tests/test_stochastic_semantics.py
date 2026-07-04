"""Robustness stochastic propagation + registry flags (issues #251, #339)."""

from __future__ import annotations

from types import SimpleNamespace

from raitap.robustness.assessors.foolbox_assessor import FoolboxAssessor
from raitap.robustness.assessors.imagecorruptions_assessor import ImageCorruptionsAssessor
from raitap.robustness.assessors.torchattacks_assessor import TorchattacksAssessor
from raitap.robustness.contracts import (
    AssessmentKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessSemantics,
    ThreatModel,
)
from raitap.robustness.semantics import AssessorAlgorithmSpec, assessor_semantics


def test_registry_stochastic_flags() -> None:
    assert TorchattacksAssessor.algorithm_registry["PGD"].stochastic is True
    assert TorchattacksAssessor.algorithm_registry["PGDL2"].stochastic is True
    assert TorchattacksAssessor.algorithm_registry["AutoAttack"].stochastic is True
    assert TorchattacksAssessor.algorithm_registry["Square"].stochastic is True
    assert TorchattacksAssessor.algorithm_registry["OnePixel"].stochastic is True
    assert TorchattacksAssessor.algorithm_registry["FGSM"].stochastic is False
    assert TorchattacksAssessor.algorithm_registry["CW"].stochastic is False

    assert FoolboxAssessor.algorithm_registry["LinfPGD"].stochastic is True
    assert FoolboxAssessor.algorithm_registry["BoundaryAttack"].stochastic is True
    assert FoolboxAssessor.algorithm_registry["LinfFastGradientAttack"].stochastic is False

    # imagecorruptions: every corruption is statistical-sampling -> stochastic.
    assert all(hints.stochastic for hints in ImageCorruptionsAssessor.algorithm_registry.values())


def test_assessor_semantics_carries_stochastic() -> None:
    assessor = TorchattacksAssessor(algorithm="PGD", eps=0.03, steps=10)
    inputs = SimpleNamespace(shape=(1, 3, 8, 8))

    semantics = assessor_semantics(
        assessor,
        call_kwargs={},
        raitap_kwargs={},
        inputs=inputs,
        targets=None,
    )

    assert semantics.stochastic is True


def test_assessor_semantics_deterministic_attack_is_false() -> None:
    assessor = TorchattacksAssessor(algorithm="FGSM", eps=0.03)
    inputs = SimpleNamespace(shape=(1, 3, 8, 8))

    semantics = assessor_semantics(
        assessor,
        call_kwargs={},
        raitap_kwargs={},
        inputs=inputs,
        targets=None,
    )

    assert semantics.stochastic is False


def test_assessor_spec_seeding_drives_stochastic_property() -> None:
    spec = AssessorAlgorithmSpec(
        assessment_kind=AssessmentKind.EMPIRICAL_ATTACK,
        threat_model=ThreatModel.WHITE_BOX,
        objective=Objective.UNTARGETED,
        seeding="global_rng",
    )

    assert spec.stochastic is True


def test_robustness_semantics_carries_seeding() -> None:
    semantics = RobustnessSemantics(
        assessment_kind=AssessmentKind.EMPIRICAL_ATTACK,
        threat_model=ThreatModel.WHITE_BOX,
        objective=Objective.UNTARGETED,
        families=frozenset({"gradient_sign"}),
        perturbation=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.03),
        seeding="global_rng",
    )
    assert semantics.seeding == "global_rng"
    assert semantics.stochastic is True
