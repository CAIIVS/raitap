from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from raitap.robustness.contracts import (
    VERDICT_CODES,
    AssessmentKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessSemantics,
    RobustnessVerdict,
    ThreatModel,
    decode_verdict,
    encode_verdict,
)


def test_verdict_codes_are_unique_and_round_trip() -> None:
    codes = list(VERDICT_CODES.values())
    assert len(codes) == len(set(codes))
    for verdict in RobustnessVerdict:
        assert decode_verdict(encode_verdict(verdict)) == verdict


def test_decode_unknown_code_raises() -> None:
    with pytest.raises(ValueError):
        decode_verdict(999)


def test_robustness_semantics_is_frozen() -> None:
    semantics = RobustnessSemantics(
        assessment_kind=AssessmentKind.EMPIRICAL_ATTACK,
        threat_model=ThreatModel.WHITE_BOX,
        objective=Objective.UNTARGETED,
        families=frozenset({"gradient_sign"}),
        perturbation=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.03),
    )
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        semantics.objective = Objective.TARGETED  # type: ignore[misc]


def test_perturbation_budget_serialisable() -> None:
    budget = PerturbationBudget(norm=PerturbationNorm.L2, epsilon=0.5, step_size=0.01, steps=20)
    payload = {
        "norm": budget.norm.value,
        "epsilon": budget.epsilon,
        "step_size": budget.step_size,
        "steps": budget.steps,
    }
    assert json.loads(json.dumps(payload)) == payload


def test_statistical_sampling_kind_maps_to_average_case() -> None:
    from raitap.robustness.contracts import (
        AssessmentKind,
        RobustnessCase,
        case_for,
    )

    assert case_for(AssessmentKind.STATISTICAL_SAMPLING) is RobustnessCase.AVERAGE_CASE
    assert case_for(AssessmentKind.EMPIRICAL_ATTACK) is RobustnessCase.WORST_CASE
    assert case_for(AssessmentKind.FORMAL_VERIFICATION) is RobustnessCase.WORST_CASE


def test_every_assessment_kind_has_a_case() -> None:
    from raitap.robustness.contracts import AssessmentKind, case_for

    for kind in AssessmentKind:
        case_for(kind)  # raises KeyError if a kind is unmapped


def test_new_verdicts_have_stable_codes() -> None:
    from raitap.robustness.contracts import (
        VERDICT_CODES,
        RobustnessVerdict,
        decode_verdict,
        encode_verdict,
    )

    assert VERDICT_CODES[RobustnessVerdict.CORRECT_UNDER_PERTURBATION] == 7
    assert VERDICT_CODES[RobustnessVerdict.MISCLASSIFIED_UNDER_PERTURBATION] == 8
    for verdict in (
        RobustnessVerdict.CORRECT_UNDER_PERTURBATION,
        RobustnessVerdict.MISCLASSIFIED_UNDER_PERTURBATION,
    ):
        assert decode_verdict(encode_verdict(verdict)) is verdict


def test_perturbation_distribution_is_a_perturbation_region() -> None:
    from raitap.robustness.contracts import PerturbationDistribution, PerturbationRegion

    dist = PerturbationDistribution(corruption_name="gaussian_noise", severity=3)
    assert isinstance(dist, PerturbationRegion)
    assert dist.corruption_name == "gaussian_noise"
    assert dist.severity == 3


def test_threat_model_not_applicable_exists() -> None:
    from raitap.robustness.contracts import ThreatModel

    assert ThreatModel.NOT_APPLICABLE.value == "not_applicable"


def test_semantics_case_property_is_derived() -> None:
    from raitap.robustness.contracts import (
        AssessmentKind,
        Objective,
        PerturbationDistribution,
        RobustnessCase,
        RobustnessSemantics,
        ThreatModel,
    )

    semantics = RobustnessSemantics(
        assessment_kind=AssessmentKind.STATISTICAL_SAMPLING,
        threat_model=ThreatModel.NOT_APPLICABLE,
        objective=Objective.UNTARGETED,
        families=frozenset({"common_corruption"}),
        perturbation=PerturbationDistribution(corruption_name="fog", severity=2),
    )
    assert semantics.case is RobustnessCase.AVERAGE_CASE
