"""Decorator integration test: a decorated stub assessor must land in _BUILDERS
under the robustness group."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from raitap.robustness.assessors.base_assessor import EmpiricalAttackAssessor
from raitap.robustness.assessors.registration import register_robustness_adapter
from raitap.robustness.contracts import (
    MethodKind,
    Objective,
    PerturbationNorm,
    ThreatModel,
)
from raitap.robustness.semantics import AssessorSemanticsHints

if TYPE_CHECKING:
    from torch import nn


def test_register_robustness_adapter_registers_under_robustness_group() -> None:
    @register_robustness_adapter(
        registry_name="_stub_attack",
        extra="_stub_extra",
        library="_stub_lib",
        algorithm_registry={
            "_stub_alg": AssessorSemanticsHints(
                MethodKind.EMPIRICAL_ATTACK,
                ThreatModel.WHITE_BOX,
                Objective.UNTARGETED,
                PerturbationNorm.LINF,
                families=frozenset({"stub"}),
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
