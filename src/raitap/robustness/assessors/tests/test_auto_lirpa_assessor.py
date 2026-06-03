"""Unit tests for :class:`AutoLiRPAAssessor` (auto_LiRPA fully mocked)."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from raitap.robustness.assessors import AutoLiRPAAssessor
from raitap.robustness.contracts import (
    PerturbationBudget,
    PerturbationNorm,
    RobustnessVerdict,
)
from raitap.robustness.exceptions import AssessorBackendIncompatibilityError
from raitap.types import Capability

# ---------------------------------------------------------------------------
# auto_LiRPA fake module
# ---------------------------------------------------------------------------


@dataclass
class _LirpaState:
    """Records what the adapter passed to the (faked) auto_LiRPA classes."""

    lower: torch.Tensor = field(default_factory=lambda: torch.tensor([[2.0, 0.0, 0.0]]))
    upper: torch.Tensor = field(default_factory=lambda: torch.tensor([[3.0, 1.0, 1.0]]))
    method: str | None = None
    norm: float | None = None
    eps: float | None = None
    constructed_with: list[Any] = field(default_factory=list)
    compute_bounds_error: Exception | None = None


@pytest.fixture
def fake_lirpa(monkeypatch: pytest.MonkeyPatch) -> _LirpaState:
    """Install a stub ``auto_LiRPA`` package with controllable bounds."""
    state = _LirpaState()

    class _FakeBoundedModule:
        def __init__(self, model: Any, example: Any) -> None:
            state.constructed_with.append((model, example))

        def compute_bounds(self, x: Any, method: str = "CROWN") -> tuple[Any, Any]:
            state.method = method
            if state.compute_bounds_error is not None:
                raise state.compute_bounds_error
            return state.lower, state.upper

    class _FakePerturbationLpNorm:
        def __init__(self, norm: float, eps: float) -> None:
            state.norm = norm
            state.eps = eps

    class _FakeBoundedTensor:
        def __init__(self, tensor: Any, perturbation: Any) -> None:
            self.tensor = tensor
            self.perturbation = perturbation

    module = types.ModuleType("auto_LiRPA")
    module.BoundedModule = _FakeBoundedModule  # type: ignore[attr-defined]
    module.BoundedTensor = _FakeBoundedTensor  # type: ignore[attr-defined]
    module.PerturbationLpNorm = _FakePerturbationLpNorm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "auto_LiRPA", module)
    return state


class _IdentityModel(torch.nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 3) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1) if x.ndim > 2 else x)


class _TorchBackend:
    provides = frozenset({Capability.AUTOGRAD})

    def __init__(self, device_type: str = "cpu") -> None:
        self.device = torch.device(device_type)


class _OnnxBackend:
    provides = frozenset()


def _linf_budget(eps: float = 0.05) -> PerturbationBudget:
    return PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=eps)


# ---------------------------------------------------------------------------
# Registry / construction
# ---------------------------------------------------------------------------


def test_registry_contains_crown() -> None:
    assert "crown" in AutoLiRPAAssessor.algorithm_registry
    hints = AutoLiRPAAssessor.algorithm_registry["crown"]
    assert hints.norm == PerturbationNorm.LINF
    assert "bound-propagation" in hints.families
    assert "sound" in hints.families


def test_registry_l2_algorithm_is_l2() -> None:
    assert AutoLiRPAAssessor.algorithm_registry["crown-l2"].norm == PerturbationNorm.L2


def test_constructor_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unknown algorithm"):
        AutoLiRPAAssessor(algorithm="bogus")


def test_default_algorithm_is_crown() -> None:
    assert AutoLiRPAAssessor().algorithm == "crown"


# ---------------------------------------------------------------------------
# verify_sample — verdict mapping
# ---------------------------------------------------------------------------


def test_verify_sample_target_dominates_maps_to_verified(fake_lirpa: _LirpaState) -> None:
    # lb[0]=2.0 > max(ub[1], ub[2]) = 1.0 → VERIFIED.
    fake_lirpa.lower = torch.tensor([[2.0, 0.0, 0.0]])
    fake_lirpa.upper = torch.tensor([[3.0, 1.0, 1.0]])
    assessor = AutoLiRPAAssessor(algorithm="crown")

    outcome = assessor.verify_sample(
        _IdentityModel(),
        torch.zeros(1, 3),
        torch.tensor([0]),
        budget=_linf_budget(),
    )

    assert outcome.verdict == RobustnessVerdict.VERIFIED
    assert outcome.counter_example is None
    assert outcome.lower_bounds is not None and outcome.upper_bounds is not None
    assert tuple(outcome.lower_bounds.shape) == (3,)
    assert outcome.lower_bounds.dtype == torch.float32
    assert torch.allclose(outcome.lower_bounds, torch.tensor([2.0, 0.0, 0.0]))


def test_verify_sample_overlap_maps_to_unknown_with_bounds(fake_lirpa: _LirpaState) -> None:
    # lb[0]=0.5 < max(ub[others])=1.0 → UNKNOWN, but bounds still populated.
    fake_lirpa.lower = torch.tensor([[0.5, 0.0, 0.0]])
    fake_lirpa.upper = torch.tensor([[1.5, 1.0, 1.0]])
    assessor = AutoLiRPAAssessor()

    outcome = assessor.verify_sample(
        _IdentityModel(),
        torch.zeros(1, 3),
        torch.tensor([0]),
        budget=_linf_budget(),
    )

    assert outcome.verdict == RobustnessVerdict.UNKNOWN
    assert outcome.lower_bounds is not None
    assert outcome.upper_bounds is not None
    assert torch.allclose(outcome.upper_bounds, torch.tensor([1.5, 1.0, 1.0]))


def test_verify_sample_never_falsified(fake_lirpa: _LirpaState) -> None:
    """Sound + incomplete: even a clearly non-robust point is UNKNOWN, not FALSIFIED."""
    fake_lirpa.lower = torch.tensor([[-5.0, 0.0, 0.0]])
    fake_lirpa.upper = torch.tensor([[-4.0, 9.0, 9.0]])
    outcome = AutoLiRPAAssessor().verify_sample(
        _IdentityModel(),
        torch.zeros(1, 3),
        torch.tensor([0]),
        budget=_linf_budget(),
    )
    assert outcome.verdict == RobustnessVerdict.UNKNOWN
    assert outcome.verdict != RobustnessVerdict.FALSIFIED


def test_verify_sample_rejects_norm_algorithm_mismatch(fake_lirpa: _LirpaState) -> None:
    # ``crown`` is L∞; an L2 budget must raise rather than silently run L2.
    with pytest.raises(ValueError, match=r"Linf|L2"):
        AutoLiRPAAssessor(algorithm="crown").verify_sample(
            _IdentityModel(),
            torch.zeros(1, 3),
            torch.tensor([0]),
            budget=PerturbationBudget(norm=PerturbationNorm.L2, epsilon=0.1),
        )


def test_overlapping_maxpool_error_is_rewritten(fake_lirpa: _LirpaState) -> None:
    # auto-LiRPA raises a cryptic ValueError on overlapping MaxPool (e.g. ResNet);
    # the decorator's error_patterns rewrite it into an actionable message.
    fake_lirpa.compute_bounds_error = ValueError(
        "self.stride ([2, 2]) != self.kernel_size ([3, 3])"
    )
    with pytest.raises(Exception, match="non-overlapping pooling"):
        AutoLiRPAAssessor(algorithm="crown").verify_sample(
            _IdentityModel(),
            torch.zeros(1, 3),
            torch.tensor([0]),
            budget=_linf_budget(),
        )


def test_verify_sample_rejects_out_of_range_target(fake_lirpa: _LirpaState) -> None:
    with pytest.raises(ValueError, match="out of range"):
        AutoLiRPAAssessor().verify_sample(
            _IdentityModel(),
            torch.zeros(1, 3),
            torch.tensor([9]),
            budget=_linf_budget(),
        )


# ---------------------------------------------------------------------------
# Method + norm wiring
# ---------------------------------------------------------------------------


def test_linf_algorithm_passes_inf_norm_and_crown_method(fake_lirpa: _LirpaState) -> None:
    AutoLiRPAAssessor(algorithm="crown").verify_sample(
        _IdentityModel(),
        torch.zeros(1, 3),
        torch.tensor([0]),
        budget=_linf_budget(eps=0.1),
    )
    assert fake_lirpa.method == "CROWN"
    assert fake_lirpa.norm == float("inf")
    assert fake_lirpa.eps == pytest.approx(0.1)


def test_ibp_algorithm_passes_ibp_method(fake_lirpa: _LirpaState) -> None:
    AutoLiRPAAssessor(algorithm="ibp").verify_sample(
        _IdentityModel(),
        torch.zeros(1, 3),
        torch.tensor([0]),
        budget=_linf_budget(),
    )
    assert fake_lirpa.method == "IBP"


def test_l2_algorithm_passes_norm_2(fake_lirpa: _LirpaState) -> None:
    AutoLiRPAAssessor(algorithm="crown-l2").verify_sample(
        _IdentityModel(),
        torch.zeros(1, 3),
        torch.tensor([0]),
        budget=PerturbationBudget(norm=PerturbationNorm.L2, epsilon=0.25),
    )
    assert fake_lirpa.method == "CROWN"
    assert fake_lirpa.norm == 2.0
    assert fake_lirpa.eps == pytest.approx(0.25)


def test_epsilon_falls_back_to_constructor_when_budget_eps_none(fake_lirpa: _LirpaState) -> None:
    AutoLiRPAAssessor(algorithm="crown", epsilon=0.07).verify_sample(
        _IdentityModel(),
        torch.zeros(1, 3),
        torch.tensor([0]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=None),
    )
    assert fake_lirpa.eps == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# Backend compatibility + lazy import
# ---------------------------------------------------------------------------


def test_check_backend_compat_rejects_onnx_backend() -> None:
    with pytest.raises(AssessorBackendIncompatibilityError, match="autograd"):
        AutoLiRPAAssessor().check_backend_compat(_OnnxBackend())


def test_check_backend_compat_accepts_torch_backend() -> None:
    AutoLiRPAAssessor().check_backend_compat(_TorchBackend("cpu"))  # no raise.


def test_check_backend_compat_warns_on_intel_xpu() -> None:
    assessor = AutoLiRPAAssessor()
    with pytest.warns(UserWarning, match="Intel GPUs"):
        assessor.check_backend_compat(_TorchBackend("xpu"))  # autograd True → no raise.


def test_verify_sample_raises_clear_import_error_without_auto_lirpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    real_import = importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "auto_LiRPA":
            raise ModuleNotFoundError("No module named 'auto_LiRPA'")
        return real_import(name, *args, **kwargs)

    # ``AdapterMixin._lazy_import`` calls ``importlib.import_module`` and rewrites
    # ``ModuleNotFoundError`` into a friendly install-hint ImportError.
    monkeypatch.setattr("raitap._adapters.importlib.import_module", fake_import)
    with pytest.raises(ImportError, match="auto-lirpa"):
        AutoLiRPAAssessor().verify_sample(
            _IdentityModel(),
            torch.zeros(1, 3),
            torch.tensor([0]),
            budget=_linf_budget(),
        )
