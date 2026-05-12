"""Unit tests for :class:`MarabouAssessor` (maraboupy fully mocked)."""

from __future__ import annotations

import math
import sys
import types
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np
import pytest
import torch

from raitap.robustness.assessors import MarabouAssessor
from raitap.robustness.assessors.marabou_assessor import _bisect_output_bound
from raitap.robustness.contracts import (
    PerturbationBudget,
    PerturbationNorm,
    RobustnessVerdict,
)
from raitap.robustness.exceptions import AssessorBackendIncompatibilityError

if TYPE_CHECKING:
    from pathlib import Path

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
        self.equations: list[Any] = []
        self.solve_results: list[tuple[str, dict[int, float], object]] = []
        self.solve_calls: list[dict[str, Any]] = []
        # Back-compat default still used by existing tests:
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

    def addEquation(self, equation: Any) -> None:  # noqa: N802 — Marabou API
        self.equations.append(equation)

    def solve(self, options: object | None = None) -> tuple[str, dict[int, float], object]:
        del options
        self.solve_calls.append(
            {
                "lower_bounds": dict(self.lower_bounds),
                "upper_bounds": dict(self.upper_bounds),
            }
        )
        if self.solve_results:
            return self.solve_results.pop(0)
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
    # 5 output classes minus the target class → 4 per-class solves at 0.5s each.
    assert outcome.runtime_seconds == pytest.approx(2.0)
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
    assert assessor.init_kwargs["bound_search_range"] == 50.0
    assert assessor.init_kwargs["bound_tolerance"] == 0.05


def test_bisect_output_bound_lower_converges_to_true_minimum(
    fake_maraboupy: _FakeNetwork, tmp_path: Any
) -> None:
    """Mock returns UNSAT iff probe `c` is below the true min (0.3)."""
    from maraboupy import Marabou  # type: ignore[import-not-found]

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    true_min = 0.3
    flat_sample = np.zeros(5, dtype=np.float32)

    def scripted_solve(options: object | None = None) -> tuple[str, dict[int, float], object]:
        del options
        mid = fake_maraboupy.upper_bounds[5]
        if mid < true_min:
            return ("unsat", {}, _FakeStats(0.0))
        return ("sat", {}, _FakeStats(0.0))

    fake_maraboupy.solve = scripted_solve  # type: ignore[method-assign]

    bound = _bisect_output_bound(
        onnx_path=onnx_path,
        flat_sample=flat_sample,
        eps=0.05,
        output_index=0,
        mode="lower",
        search_range=1.0,
        tolerance=1e-3,
        timeout_s=1.0,
    )
    assert abs(bound - true_min) <= 1e-3
    assert Marabou.read_onnx.call_count >= 1


def test_bisect_output_bound_upper_converges_to_true_maximum(
    fake_maraboupy: _FakeNetwork, tmp_path: Any
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    true_max = -0.2
    flat_sample = np.zeros(5, dtype=np.float32)

    def scripted_solve(options: object | None = None) -> tuple[str, dict[int, float], object]:
        del options
        mid = fake_maraboupy.lower_bounds[5]
        if mid > true_max:
            return ("unsat", {}, _FakeStats(0.0))
        return ("sat", {}, _FakeStats(0.0))

    fake_maraboupy.solve = scripted_solve  # type: ignore[method-assign]

    bound = _bisect_output_bound(
        onnx_path=onnx_path,
        flat_sample=flat_sample,
        eps=0.05,
        output_index=0,
        mode="upper",
        search_range=1.0,
        tolerance=1e-3,
        timeout_s=1.0,
    )
    assert abs(bound - true_max) <= 1e-3


def test_bisect_output_bound_stops_conservatively_on_unknown_exit_code(
    fake_maraboupy: _FakeNetwork, tmp_path: Path
) -> None:
    """TIMEOUT/UNKNOWN must NOT collapse to SAT — stop with current conservative bound."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    flat_sample = np.zeros(5, dtype=np.float32)

    def scripted_solve(options: object | None = None) -> tuple[str, dict[int, float], object]:
        del options
        return ("TIMEOUT", {}, _FakeStats(0.0))

    fake_maraboupy.solve = scripted_solve  # type: ignore[method-assign]

    bound = _bisect_output_bound(
        onnx_path=onnx_path,
        flat_sample=flat_sample,
        eps=0.05,
        output_index=0,
        mode="lower",
        search_range=1.0,
        tolerance=1e-3,
        timeout_s=1.0,
    )
    # With every probe TIMEOUT no UNSAT ever certifies lo, so the bisect must
    # return NaN rather than the unproven sentinel.
    assert math.isnan(bound)


def test_bisect_output_bound_warns_when_all_probes_inconclusive(
    fake_maraboupy: _FakeNetwork,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    flat_sample = np.zeros(5, dtype=np.float32)

    def scripted_solve(options: object | None = None) -> tuple[str, dict[int, float], object]:
        del options
        return ("TIMEOUT", {}, _FakeStats(0.0))

    fake_maraboupy.solve = scripted_solve  # type: ignore[method-assign]

    with caplog.at_level("WARNING", logger="raitap.robustness.assessors.marabou_assessor"):
        _bisect_output_bound(
            onnx_path=onnx_path,
            flat_sample=flat_sample,
            eps=0.05,
            output_index=2,
            mode="lower",
            search_range=1.0,
            tolerance=1e-3,
            timeout_s=1.0,
        )

    assert any("no certifying UNSAT" in rec.message for rec in caplog.records)
    assert any("output_index=2" in rec.message for rec in caplog.records)


def test_bisect_output_bound_returns_nan_when_no_unsat_observed(
    fake_maraboupy: _FakeNetwork, tmp_path: Path
) -> None:
    """SAT-only probes leave the bound uncertified → must return NaN."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    flat_sample = np.zeros(5, dtype=np.float32)

    def scripted_solve(options: object | None = None) -> tuple[str, dict[int, float], object]:
        del options
        return ("sat", {}, _FakeStats(0.0))

    fake_maraboupy.solve = scripted_solve  # type: ignore[method-assign]

    bound = _bisect_output_bound(
        onnx_path=onnx_path,
        flat_sample=flat_sample,
        eps=0.05,
        output_index=0,
        mode="lower",
        search_range=1.0,
        tolerance=1e-3,
        timeout_s=1.0,
    )
    assert math.isnan(bound)


def test_verify_sample_returns_none_bounds_when_flag_disabled(
    fake_maraboupy: _FakeNetwork, tmp_path: Any
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    fake_maraboupy.solve_result = ("unsat", {}, _FakeStats(0.0))

    class _Backend:
        def __init__(self, p: Any) -> None:
            self.onnx_path = p

    assessor = MarabouAssessor(compute_output_bounds=False)
    outcome = assessor.verify_sample(
        model=_IdentityModel(),
        sample=torch.zeros(1, 5),
        target=torch.tensor([0]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        backend=_Backend(onnx_path),
    )
    assert outcome.verdict == RobustnessVerdict.VERIFIED
    assert outcome.lower_bounds is None
    assert outcome.upper_bounds is None


def test_verify_sample_populates_bounds_when_flag_enabled_and_verified(
    fake_maraboupy: _FakeNetwork, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    fake_maraboupy.solve_result = ("unsat", {}, _FakeStats(0.0))

    fake_bounds = [(-0.1 * k, 0.1 * k) for k in range(5)]
    calls: list[tuple[int, str]] = []

    def fake_bisect(**kwargs: Any) -> float:
        calls.append((int(kwargs["output_index"]), str(kwargs["mode"])))
        lo, hi = fake_bounds[int(kwargs["output_index"])]
        return lo if kwargs["mode"] == "lower" else hi

    monkeypatch.setattr(
        "raitap.robustness.assessors.marabou_assessor._bisect_output_bound",
        fake_bisect,
    )

    class _Backend:
        def __init__(self, p: Any) -> None:
            self.onnx_path = p

    assessor = MarabouAssessor(compute_output_bounds=True)
    outcome = assessor.verify_sample(
        model=_IdentityModel(),
        sample=torch.zeros(1, 5),
        target=torch.tensor([0]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        backend=_Backend(onnx_path),
    )
    assert outcome.lower_bounds is not None and outcome.upper_bounds is not None
    assert outcome.lower_bounds.shape == (5,)
    assert outcome.upper_bounds.shape == (5,)
    assert outcome.lower_bounds.dtype == torch.float32
    assert torch.allclose(outcome.lower_bounds, torch.tensor([lo for lo, _ in fake_bounds]))
    assert torch.allclose(outcome.upper_bounds, torch.tensor([hi for _, hi in fake_bounds]))
    assert sorted(calls) == sorted(
        [(k, "lower") for k in range(5)] + [(k, "upper") for k in range(5)]
    )


def test_verify_sample_skips_bounds_for_falsified_verdict(
    fake_maraboupy: _FakeNetwork, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    fake_maraboupy.solve_result = (
        "sat",
        {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        _FakeStats(0.0),
    )

    def boom(**kwargs: Any) -> float:
        raise AssertionError("bisect must not be called on FALSIFIED")

    monkeypatch.setattr("raitap.robustness.assessors.marabou_assessor._bisect_output_bound", boom)

    class _Backend:
        def __init__(self, p: Any) -> None:
            self.onnx_path = p

    assessor = MarabouAssessor(compute_output_bounds=True)
    outcome = assessor.verify_sample(
        model=_IdentityModel(),
        sample=torch.zeros(1, 5),
        target=torch.tensor([0]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        backend=_Backend(onnx_path),
    )
    assert outcome.verdict == RobustnessVerdict.FALSIFIED
    assert outcome.lower_bounds is None
    assert outcome.upper_bounds is None


def test_assess_propagates_output_bounds_to_result(
    fake_maraboupy: _FakeNetwork, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two samples: first VERIFIED with bounds, second FALSIFIED → NaN row."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    # Per-class solve loop issues N-1 calls per sample (N=5 outputs). Sample 0
    # needs 4 UNSAT to reach VERIFIED; sample 1 short-circuits on first SAT.
    fake_maraboupy.solve_results = [
        *[("unsat", {}, _FakeStats(0.0))] * 4,  # sample 0 → VERIFIED
        ("sat", dict.fromkeys(range(5), 0.0), _FakeStats(0.0)),  # sample 1 → FALSIFIED
    ]

    monkeypatch.setattr(
        "raitap.robustness.assessors.marabou_assessor._bisect_output_bound",
        lambda **kw: -1.0 if kw["mode"] == "lower" else 1.0,
    )

    class _Backend:
        def __init__(self, p: Any) -> None:
            self.onnx_path = p

    assessor = MarabouAssessor(compute_output_bounds=True)
    result = assessor.assess(
        model=_IdentityModel(),
        inputs=torch.zeros(2, 5),
        targets=torch.tensor([0, 0]),
        backend=_Backend(onnx_path),
    )
    assert result.output_bounds is not None
    lower = result.output_bounds["lower"]
    upper = result.output_bounds["upper"]
    assert lower.shape == (2, 5)
    assert torch.allclose(lower[0], torch.full((5,), -1.0))
    assert torch.allclose(upper[0], torch.full((5,), 1.0))
    assert torch.isnan(lower[1]).all()
    assert torch.isnan(upper[1]).all()


def test_bisect_output_bound_rejects_non_positive_tolerance(tmp_path: Path) -> None:
    flat_sample = np.zeros(5, dtype=np.float32)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="must be > 0"):
        _bisect_output_bound(
            onnx_path=onnx_path,
            flat_sample=flat_sample,
            eps=0.05,
            output_index=0,
            mode="lower",
            search_range=1.0,
            tolerance=0.0,
            timeout_s=1.0,
        )
