from __future__ import annotations

import pytest

from raitap.models.access import AutogradModelProvider, explanation_model
from raitap.types import Capability
from raitap.utils.errors import BackendIncompatibilityError


class _FakeAdapter:
    def __init__(self, requires: frozenset[Capability]) -> None:
        self._requires = requires

    def required_capabilities(self) -> frozenset[Capability]:
        return self._requires


class _CallableBackend:
    provides = frozenset()

    def __call__(self, x):  # noqa: ANN001
        return x

    def predict_callable(self):  # noqa: ANN202
        return self.__call__


class _AutogradBackend(_CallableBackend):
    provides = frozenset({Capability.AUTOGRAD})

    def autograd_module(self):  # noqa: ANN202
        return "the-module"


def test_model_agnostic_adapter_gets_predict_callable() -> None:
    backend = _CallableBackend()
    model = explanation_model(backend, _FakeAdapter(frozenset()))
    assert model == backend.__call__


def test_autograd_adapter_gets_autograd_module() -> None:
    model = explanation_model(_AutogradBackend(), _FakeAdapter(frozenset({Capability.AUTOGRAD})))
    assert model == "the-module"


def test_autograd_provider_protocol_is_runtime_checkable() -> None:
    assert isinstance(_AutogradBackend(), AutogradModelProvider)
    assert not isinstance(_CallableBackend(), AutogradModelProvider)


def test_declared_capability_without_impl_raises() -> None:
    class _Broken(_CallableBackend):  # claims AUTOGRAD but has no autograd_module
        provides = frozenset({Capability.AUTOGRAD})

    with pytest.raises(BackendIncompatibilityError):
        explanation_model(_Broken(), _FakeAdapter(frozenset({Capability.AUTOGRAD})))
