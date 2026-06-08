from __future__ import annotations

import numpy as np
import torch

from raitap.models.access import EstimatorProvider
from raitap.models.tree_backend import TabularTreeBackend
from raitap.types import Capability, TaskKind


class _FakeProbaEstimator:
    """Two-class estimator: probability of class 1 = sigmoid(sum(features))."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        p1 = 1.0 / (1.0 + np.exp(-X.sum(axis=1)))
        return np.stack([1.0 - p1, p1], axis=1)


class _StubTreeBackend(TabularTreeBackend):
    provides = frozenset({Capability.TREE_MODEL, Capability.PREDICT_PROBA})

    @classmethod
    def from_path(cls, path, *, model_cfg, hardware, allow_unsafe_pickle=False):  # noqa: ANN001
        raise NotImplementedError

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return self._estimator.predict_proba(X)


def _backend() -> _StubTreeBackend:
    return _StubTreeBackend(_FakeProbaEstimator())


def test_call_returns_n_by_c_probability_tensor() -> None:
    backend = _backend()
    inputs = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    out = backend(inputs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2)
    assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-5)


def test_fitted_estimator_returns_raw_estimator() -> None:
    estimator = _FakeProbaEstimator()
    backend = _StubTreeBackend(estimator)
    assert backend.fitted_estimator() is estimator


def test_implements_estimator_provider() -> None:
    assert isinstance(_backend(), EstimatorProvider)


def test_predict_callable_is_call() -> None:
    backend = _backend()
    assert backend.predict_callable() == backend.__call__


def test_defaults_cpu_classification() -> None:
    backend = _backend()
    assert backend.hardware_label == "CPU"
    assert backend.task_kind == TaskKind.classification
