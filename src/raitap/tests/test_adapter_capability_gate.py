from typing import ClassVar

import pytest

from raitap._adapters import AdapterMixin
from raitap.types import Capability
from raitap.utils.errors import BackendIncompatibilityError


class _Backend:
    def __init__(self, provides: frozenset[Capability]) -> None:
        self.provides = provides


class _Adapter(AdapterMixin):
    algorithm_registry: ClassVar = {
        "needs_grad": type("H", (), {"requires": frozenset({Capability.AUTOGRAD})})()
    }

    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm


def test_gate_passes_when_capabilities_met() -> None:
    _Adapter("needs_grad").check_backend_compat(
        _Backend(frozenset({Capability.AUTOGRAD}))
    )  # no raise


def test_gate_rejects_when_capability_missing() -> None:
    with pytest.raises(BackendIncompatibilityError, match="autograd"):
        _Adapter("needs_grad").check_backend_compat(_Backend(frozenset()))


def test_empty_requires_runs_anywhere() -> None:
    class _Agnostic(AdapterMixin):
        algorithm_registry: ClassVar = {"x": type("H", (), {"requires": frozenset()})()}

        def __init__(self) -> None:
            self.algorithm = "x"

    _Agnostic().check_backend_compat(_Backend(frozenset()))  # no raise
