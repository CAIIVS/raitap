"""Decorator integration test: a decorated stub assessor must land in _BUILDERS
under the robustness group."""

from __future__ import annotations

from raitap.robustness.assessors.base_assessor import EmpiricalAttackAssessor
from raitap.robustness.assessors.registration import register_robustness_adapter


def test_register_robustness_adapter_registers_under_robustness_group() -> None:
    @register_robustness_adapter(
        registry_name="_stub_attack",
        extra="_stub_extra",
        library="_stub_lib",
    )
    # abstract=True skips both WithAlgorithmRegistry and AdapterMixin pre-validation
    # / registration. The decorator is the sole registrar — assertions only pass if
    # `register_robustness_adapter` actually ran.
    class _StubAssessor(EmpiricalAttackAssessor, abstract=True):
        def __init__(self, algorithm: str):
            super().__init__()
            self.algorithm = algorithm

        def check_backend_compat(self, backend: object) -> None:
            del backend

        def generate_adversarial(self, model, inputs, targets, *, backend=None, **kw):
            del model, inputs, targets, backend, kw
            return None  # type: ignore[return-value]

    from raitap._adapters import ADAPTER_EXTRAS, _BUILDERS

    assert "_stub_attack" in _BUILDERS["robustness"]
    assert ADAPTER_EXTRAS["_StubAssessor"] == "_stub_extra"
