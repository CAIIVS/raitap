"""Decorator integration test: a decorated stub assessor must land in _BUILDERS
under the robustness group."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from raitap import adapters
from raitap.robustness.assessors.base_assessor import EmpiricalAttackAssessor
from raitap.robustness.contracts import (
    AssessmentKind,
    Objective,
    PerturbationNorm,
    ThreatModel,
)
from raitap.robustness.semantics import AssessorSemanticsHints

if TYPE_CHECKING:
    from torch import nn


def test_robustness_adapter_registers_under_robustness_group() -> None:
    @adapters.robustness(
        registry_name="_stub_attack",
        extra="_stub_extra",
        library="_stub_lib",
        algorithm_registry={
            "_stub_alg": AssessorSemanticsHints(
                AssessmentKind.EMPIRICAL_ATTACK,
                ThreatModel.WHITE_BOX,
                Objective.UNTARGETED,
                PerturbationNorm.LINF,
                families={"stub"},
            ),
        },
    )
    class _StubAssessor(EmpiricalAttackAssessor):
        def __init__(self, algorithm: str):
            super().__init__()
            self.algorithm = algorithm

        def check_backend_compat(self, backend: object) -> None:
            del backend

        def generate_adversarial(
            self,
            model: nn.Module,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            *,
            backend: object | None = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            del model, inputs, targets, backend, kwargs
            return torch.zeros(0)

    from raitap._adapters import _BUILDERS, ADAPTER_EXTRAS

    assert "_stub_attack" in _BUILDERS["robustness"]
    assert ADAPTER_EXTRAS["_StubAssessor"] == "_stub_extra"


def test_budget_kwarg_source_via_decorator() -> None:
    from raitap.robustness.assessors.base_assessor import EmpiricalAttackAssessor
    from raitap.robustness.assessors.registration import robustness_adapter
    from raitap.robustness.contracts import AssessmentKind, Objective, PerturbationNorm, ThreatModel
    from raitap.robustness.semantics import AssessorSemanticsHints

    @robustness_adapter(
        registry_name="_stub_budget",
        budget_kwarg_source="call_kwargs",
        algorithm_registry={
            "x": AssessorSemanticsHints(
                AssessmentKind.EMPIRICAL_ATTACK,
                ThreatModel.WHITE_BOX,
                Objective.UNTARGETED,
                PerturbationNorm.LINF,
                families={"i"},
            ),
        },
    )
    class _Stub(EmpiricalAttackAssessor):
        def generate_adversarial(self, model, inputs, targets, *, backend=None, **kw):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202
            return inputs

    assert _Stub.budget_kwarg_source == "call_kwargs"
