from __future__ import annotations

from typing import Any, cast

import pytest

from raitap.models.access import AutogradModelProvider, EstimatorProvider, explanation_model
from raitap.types import Capability


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
    model = explanation_model(cast("Any", backend), _FakeAdapter(frozenset()))
    assert model == backend.__call__


def test_autograd_adapter_gets_autograd_module() -> None:
    model = explanation_model(
        cast("Any", _AutogradBackend()), _FakeAdapter(frozenset({Capability.AUTOGRAD}))
    )
    assert model == "the-module"


def test_autograd_provider_protocol_is_runtime_checkable() -> None:
    assert isinstance(_AutogradBackend(), AutogradModelProvider)
    assert not isinstance(_CallableBackend(), AutogradModelProvider)


def test_declared_capability_without_impl_raises() -> None:
    class _Broken(_CallableBackend):  # claims AUTOGRAD but has no autograd_module
        provides = frozenset({Capability.AUTOGRAD})

    with pytest.raises(TypeError):
        explanation_model(cast("Any", _Broken()), _FakeAdapter(frozenset({Capability.AUTOGRAD})))


class _EstimatorBackend(_CallableBackend):
    provides = frozenset({Capability.TREE_MODEL, Capability.PREDICT_PROBA})

    def fitted_estimator(self):  # noqa: ANN202
        return "the-estimator"


def test_tree_model_adapter_gets_fitted_estimator() -> None:
    model = explanation_model(
        cast("Any", _EstimatorBackend()), _FakeAdapter(frozenset({Capability.TREE_MODEL}))
    )
    assert model == "the-estimator"


def test_estimator_provider_protocol_is_runtime_checkable() -> None:
    assert isinstance(_EstimatorBackend(), EstimatorProvider)
    assert not isinstance(_CallableBackend(), EstimatorProvider)


def test_tree_model_without_impl_raises() -> None:
    class _Broken(_CallableBackend):  # claims TREE_MODEL but has no fitted_estimator
        provides = frozenset({Capability.TREE_MODEL})

    with pytest.raises(TypeError):
        explanation_model(cast("Any", _Broken()), _FakeAdapter(frozenset({Capability.TREE_MODEL})))
