"""Unit tests for :class:`MarabouAssessor` (maraboupy fully mocked)."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch

from raitap.robustness.assessors import MarabouAssessor
from raitap.robustness.contracts import (
    PerturbationBudget,
    PerturbationNorm,
    RobustnessVerdict,
)
from raitap.robustness.exceptions import AssessorBackendIncompatibilityError

# ---------------------------------------------------------------------------
# maraboupy fake module
# ---------------------------------------------------------------------------


class _FakeEquation:
    GE = "ge"
    LE = "le"
    EQ = "eq"

    def __init__(self, kind: str = "ge") -> None:
        self.kind = kind
        self.addends: list[tuple[float, int]] = []
        self.scalar: float | None = None

    def addAddend(self, coeff: float, var: int) -> None:  # noqa: N802 — Marabou API
        self.addends.append((float(coeff), int(var)))

    def setScalar(self, scalar: float) -> None:  # noqa: N802 — Marabou API
        self.scalar = float(scalar)


class _FakeNetwork:
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        # Variable IDs: 0..num_inputs-1 inputs, num_inputs..num_inputs+num_outputs-1 outputs.
        self.inputVars = [np.arange(num_inputs).reshape(1, num_inputs)]
        self.outputVars = [np.arange(num_inputs, num_inputs + num_outputs).reshape(1, num_outputs)]
        self.lower_bounds: dict[int, float] = {}
        self.upper_bounds: dict[int, float] = {}
        self.disjunctions: list[Any] = []
        self.solve_result: tuple[str, dict[int, float], object] = (
            "unsat",
            {},
            _FakeStats(0.123),
        )

    def setLowerBound(self, var: int, value: float) -> None:  # noqa: N802 — Marabou API
        self.lower_bounds[int(var)] = float(value)

    def setUpperBound(self, var: int, value: float) -> None:  # noqa: N802 — Marabou API
        self.upper_bounds[int(var)] = float(value)

    def addDisjunctionConstraint(self, disjuncts: Any) -> None:  # noqa: N802 — Marabou API
        self.disjunctions.append(disjuncts)

    def solve(self, options: object | None = None) -> tuple[str, dict[int, float], object]:
        del options
        return self.solve_result


class _FakeStats:
    def __init__(self, seconds: float) -> None:
        self._micros = int(seconds * 1e6)

    def getTotalTimeInMicro(self) -> int:  # noqa: N802 — Marabou API
        return self._micros


@pytest.fixture
def fake_maraboupy(monkeypatch: pytest.MonkeyPatch) -> _FakeNetwork:
    """Install a stub ``maraboupy`` package that returns a controllable network."""
    network = _FakeNetwork(num_inputs=5, num_outputs=5)

    marabou_module = types.ModuleType("maraboupy.Marabou")
    marabou_module.read_onnx = mock.MagicMock(return_value=network)  # type: ignore[attr-defined]
    marabou_module.createOptions = mock.MagicMock(return_value=object())  # type: ignore[attr-defined]

    core_module = types.ModuleType("maraboupy.MarabouCore")
    core_module.Equation = _FakeEquation  # type: ignore[attr-defined]

    package = types.ModuleType("maraboupy")
    package.Marabou = marabou_module  # type: ignore[attr-defined]
    package.MarabouCore = core_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "maraboupy", package)
    monkeypatch.setitem(sys.modules, "maraboupy.Marabou", marabou_module)
    monkeypatch.setitem(sys.modules, "maraboupy.MarabouCore", core_module)
    return network


class _IdentityModel(torch.nn.Module):
    def __init__(self, in_features: int = 5, out_features: int = 5) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1) if x.ndim > 2 else x)


class _OnnxBackend:
    """Backend stub that pretends to expose a pre-built ONNX file."""

    def __init__(self, onnx_path: str) -> None:
        self.onnx_path = onnx_path


# ---------------------------------------------------------------------------
# Registry / construction
# ---------------------------------------------------------------------------


def test_registry_contains_linf_box() -> None:
    assert "linf-box" in MarabouAssessor.algorithm_registry
    hints = MarabouAssessor.algorithm_registry["linf-box"]
    assert hints.norm == PerturbationNorm.LINF
    assert "smt" in hints.families


def test_constructor_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unknown algorithm"):
        MarabouAssessor(algorithm="bogus")


# ---------------------------------------------------------------------------
# verify_sample — verdict mapping
# ---------------------------------------------------------------------------


def _onnx_path(tmp_path: Any) -> str:
    """Create a dummy .onnx file so ``_backend_onnx_path`` accepts the backend."""
    path = tmp_path / "fake.onnx"
    path.write_bytes(b"\x00")  # contents irrelevant — read_onnx is mocked.
    return str(path)


def test_verify_sample_unsat_maps_to_verified(tmp_path: Any, fake_maraboupy: _FakeNetwork) -> None:
    fake_maraboupy.solve_result = ("unsat", {}, _FakeStats(0.5))
    assessor = MarabouAssessor(epsilon=0.01, timeout_s=10)
    backend = _OnnxBackend(_onnx_path(tmp_path))
    sample = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    target = torch.tensor([2])

    outcome = assessor.verify_sample(
        _IdentityModel(),
        sample,
        target,
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.01),
        backend=backend,
    )

    assert outcome.verdict == RobustnessVerdict.VERIFIED
    assert outcome.counter_example is None
    assert outcome.runtime_seconds == pytest.approx(0.5)
    # Bounds applied to all 5 inputs around the sample value.
    assert fake_maraboupy.lower_bounds[0] == pytest.approx(0.99)
    assert fake_maraboupy.upper_bounds[0] == pytest.approx(1.01)


def test_verify_sample_sat_reconstructs_counter_example_5d(
    tmp_path: Any, fake_maraboupy: _FakeNetwork
) -> None:
    fake_maraboupy.solve_result = (
        "sat",
        {0: 1.1, 1: 2.1, 2: 2.9, 3: 4.0, 4: 4.95},
        _FakeStats(0.8),
    )
    assessor = MarabouAssessor(epsilon=0.5)
    backend = _OnnxBackend(_onnx_path(tmp_path))
    sample = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    target = torch.tensor([0])

    outcome = assessor.verify_sample(
        _IdentityModel(),
        sample,
        target,
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.5),
        backend=backend,
    )

    assert outcome.verdict == RobustnessVerdict.FALSIFIED
    assert outcome.counter_example is not None
    assert tuple(outcome.counter_example.shape) == (1, 5)
    assert outcome.counter_example[0, 0].item() == pytest.approx(1.1)


def test_verify_sample_sat_reconstructs_counter_example_image(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Image-shaped sample (1, 1, 4, 4) — 16 inputs, 5 outputs."""
    network = _FakeNetwork(num_inputs=16, num_outputs=5)
    network.solve_result = (
        "sat",
        {i: float(i) for i in range(16)},
        _FakeStats(0.1),
    )
    marabou_module = types.ModuleType("maraboupy.Marabou")
    marabou_module.read_onnx = mock.MagicMock(return_value=network)  # type: ignore[attr-defined]
    marabou_module.createOptions = mock.MagicMock(return_value=object())  # type: ignore[attr-defined]
    core_module = types.ModuleType("maraboupy.MarabouCore")
    core_module.Equation = _FakeEquation  # type: ignore[attr-defined]
    package = types.ModuleType("maraboupy")
    package.Marabou = marabou_module  # type: ignore[attr-defined]
    package.MarabouCore = core_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "maraboupy", package)
    monkeypatch.setitem(sys.modules, "maraboupy.Marabou", marabou_module)
    monkeypatch.setitem(sys.modules, "maraboupy.MarabouCore", core_module)

    assessor = MarabouAssessor(epsilon=0.05)
    backend = _OnnxBackend(_onnx_path(tmp_path))
    sample = torch.zeros(1, 1, 4, 4)
    target = torch.tensor([1])

    outcome = assessor.verify_sample(
        _IdentityModel(in_features=16),
        sample,
        target,
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        backend=backend,
    )

    assert outcome.verdict == RobustnessVerdict.FALSIFIED
    assert outcome.counter_example is not None
    assert tuple(outcome.counter_example.shape) == (1, 1, 4, 4)


def test_verify_sample_timeout_maps_to_unknown(tmp_path: Any, fake_maraboupy: _FakeNetwork) -> None:
    fake_maraboupy.solve_result = ("TIMEOUT", {}, _FakeStats(120.0))
    assessor = MarabouAssessor(epsilon=0.01)
    backend = _OnnxBackend(_onnx_path(tmp_path))
    outcome = assessor.verify_sample(
        _IdentityModel(),
        torch.zeros(1, 5),
        torch.tensor([0]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.01),
        backend=backend,
    )
    assert outcome.verdict == RobustnessVerdict.UNKNOWN


def test_verify_sample_error_maps_to_error(tmp_path: Any, fake_maraboupy: _FakeNetwork) -> None:
    fake_maraboupy.solve_result = ("ERROR", {}, _FakeStats(0.1))
    assessor = MarabouAssessor(epsilon=0.01)
    backend = _OnnxBackend(_onnx_path(tmp_path))
    outcome = assessor.verify_sample(
        _IdentityModel(),
        torch.zeros(1, 5),
        torch.tensor([0]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.01),
        backend=backend,
    )
    assert outcome.verdict == RobustnessVerdict.ERROR


def test_verify_sample_rejects_non_linf_norm(tmp_path: Any, fake_maraboupy: _FakeNetwork) -> None:
    del fake_maraboupy
    assessor = MarabouAssessor(epsilon=0.01)
    backend = _OnnxBackend(_onnx_path(tmp_path))
    with pytest.raises(ValueError, match=r"LINF|Linf"):
        assessor.verify_sample(
            _IdentityModel(),
            torch.zeros(1, 5),
            torch.tensor([0]),
            budget=PerturbationBudget(norm=PerturbationNorm.L2, epsilon=0.01),
            backend=backend,
        )


# ---------------------------------------------------------------------------
# ONNX export caching
# ---------------------------------------------------------------------------


def test_onnx_export_cached_across_samples(
    fake_maraboupy: _FakeNetwork, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    del fake_maraboupy
    fake_path = tmp_path / "exported.onnx"
    export_calls = 0

    def stub_export(model: Any, sample: Any) -> Any:
        del model, sample
        nonlocal export_calls
        export_calls += 1
        return _write_dummy_onnx(fake_path)

    monkeypatch.setattr(
        "raitap.robustness.assessors.marabou_assessor._export_torch_to_onnx",
        stub_export,
    )

    assessor = MarabouAssessor(epsilon=0.01)
    model = _IdentityModel()
    sample = torch.zeros(1, 5)
    target = torch.tensor([0])
    budget = PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.01)

    assessor.check_backend_compat(None)  # reset per-assess() state.
    for _ in range(3):
        assessor.verify_sample(model, sample, target, budget=budget, backend=None)

    # Cache populated exactly once across the three samples.
    assert export_calls == 1
    assert len(assessor._onnx_cache) == 1


def _write_dummy_onnx(path: Any) -> Any:
    from pathlib import Path

    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")
    return p


def test_onnx_export_failure_raises_backend_incompatibility(
    fake_maraboupy: _FakeNetwork, monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_maraboupy

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("dynamic axes not supported")

    monkeypatch.setattr("torch.onnx.export", boom)

    assessor = MarabouAssessor(epsilon=0.01)
    with pytest.raises(AssessorBackendIncompatibilityError, match="static-shape MLPs"):
        assessor.verify_sample(
            _IdentityModel(),
            torch.zeros(1, 5),
            torch.tensor([0]),
            budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.01),
            backend=None,
        )


def test_check_backend_compat_accepts_torch_and_onnx(tmp_path: Any) -> None:
    assessor = MarabouAssessor()
    # Either backend type accepted — no raise.
    assessor.check_backend_compat(_OnnxBackend(_onnx_path(tmp_path)))
    assessor.check_backend_compat(object())
    assessor.check_backend_compat(None)


def test_compute_output_bounds_defaults_to_disabled() -> None:
    assessor = MarabouAssessor()
    assert assessor.compute_output_bounds is False
    assert assessor.bound_search_range == 1e3
    assert assessor.bound_tolerance == 1e-2


def test_compute_output_bounds_kwargs_round_trip() -> None:
    assessor = MarabouAssessor(
        compute_output_bounds=True,
        bound_search_range=50.0,
        bound_tolerance=0.05,
    )
    assert assessor.compute_output_bounds is True
    assert assessor.bound_search_range == 50.0
    assert assessor.bound_tolerance == 0.05
