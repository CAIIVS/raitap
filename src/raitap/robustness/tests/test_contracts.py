from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from raitap.robustness.contracts import (
    VERDICT_CODES,
    MethodKind,
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
        method_kind=MethodKind.EMPIRICAL_ATTACK,
        threat_model=ThreatModel.WHITE_BOX,
        objective=Objective.UNTARGETED,
        families=frozenset({"gradient_sign"}),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.03),
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
