"""Parity: raitap's MarabouAssessor returns the same verdict as a direct
maraboupy solve of the same ONNX network, input box, and output property.

Marabou is deterministic for a fixed query, so an identical encoding must yield
an identical verdict. This guards raitap's wrapper plumbing — ONNX loading,
input/output variable indexing, epsilon-box bound setting, the "target stays
strict argmax" disjunction, and the exit-code -> verdict mapping — against
silent corruption.

Runs only where ``maraboupy`` is importable (Linux, py3.11, ``marabou`` extra);
skips elsewhere. Reuses the ACAS Xu fixture helpers from the e2e module so the
network + sample are identical to the existing end-to-end coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch

from raitap.robustness.tests.test_e2e_marabou_acas_xu import (
    _load_acas_xu_onnx,
    _run_onnx,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.e2e, pytest.mark.parity, pytest.mark.robustness]


def _direct_marabou_verdict(
    model_path: Path, flat_sample: np.ndarray, target_idx: int, eps: float
) -> str:
    """Solve the same query raitap builds, directly via maraboupy.

    Returns ``"sat"`` or ``"unsat"`` (lower-cased exit code). Mirrors the
    encoding in ``MarabouAssessor`` (marabou_assessor.py): input box of
    ``value +/- eps`` and a disjunction asserting some non-target output is
    ``>=`` the target output (i.e. the target is no longer the strict argmax).
    """
    from maraboupy import Marabou, MarabouCore, MarabouUtils  # type: ignore[import-not-found]

    network = Marabou.read_onnx(str(model_path))
    input_vars = np.asarray(network.inputVars[0]).reshape(-1)
    output_vars = np.asarray(network.outputVars[0]).reshape(-1)

    for var_id, value in zip(input_vars, flat_sample, strict=True):
        network.setLowerBound(int(var_id), float(value) - eps)
        network.setUpperBound(int(var_id), float(value) + eps)

    target_var = int(output_vars[target_idx])
    disjuncts: list[list[Any]] = []
    for j, out_var in enumerate(output_vars):
        if j == target_idx:
            continue
        eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
        eq.addAddend(1.0, int(out_var))
        eq.addAddend(-1.0, target_var)
        eq.setScalar(0.0)
        disjuncts.append([eq])
    if disjuncts:
        network.addDisjunctionConstraint(disjuncts)

    options = Marabou.createOptions(timeoutInSeconds=120, verbosity=0)
    exit_code, _values, _stats = network.solve(options=options)
    # Normalise exactly as MarabouAssessor does so the verdicts compare cleanly.
    return str(exit_code).strip().lower()


@pytest.fixture
def acas_xu_fixture() -> tuple[Path, torch.Tensor, int]:
    pytest.importorskip("maraboupy")
    model_path = _load_acas_xu_onnx()
    sample_np = np.array([[10000.0, 0.0, 0.0, 600.0, 600.0]], dtype=np.float32)
    outputs = _run_onnx(model_path, sample_np)
    target_class = int(outputs.argmax(axis=1)[0])
    return model_path, torch.from_numpy(sample_np), target_class


@pytest.mark.parametrize(
    ("eps", "expected_exit_code"),
    [
        (1e-5, "unsat"),  # tiny box -> property holds -> VERIFIED
        (1e6, "sat"),  # whole domain -> counter-example exists -> FALSIFIED
    ],
)
def test_marabou_verdict_matches_direct_solve(
    acas_xu_fixture: tuple[Path, torch.Tensor, int],
    eps: float,
    expected_exit_code: str,
) -> None:
    from raitap.robustness.assessors import MarabouAssessor
    from raitap.robustness.contracts import (
        PerturbationBudget,
        PerturbationNorm,
        RobustnessVerdict,
    )

    model_path, sample, target_class = acas_xu_fixture
    flat_sample = sample.reshape(-1).numpy().astype(np.float32)

    # Direct, independent maraboupy solve of the same query.
    direct_exit = _direct_marabou_verdict(model_path, flat_sample, target_class, eps)
    assert direct_exit == expected_exit_code, (
        f"direct solve gave {direct_exit!r}, expected {expected_exit_code!r}"
    )

    # raitap path.
    class _OnnxBackend:
        onnx_path = str(model_path)

    assessor = MarabouAssessor(epsilon=eps, timeout_s=120.0)
    assessor.check_backend_compat(_OnnxBackend())
    outcome = assessor.verify_sample(
        torch.nn.Identity(),
        sample,
        torch.tensor([target_class]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=eps),
        backend=_OnnxBackend(),
    )

    expected_verdict = (
        RobustnessVerdict.VERIFIED if direct_exit == "unsat" else RobustnessVerdict.FALSIFIED
    )
    assert outcome.verdict == expected_verdict, (
        f"raitap verdict {outcome.verdict} disagrees with direct marabou {direct_exit!r}: "
        f"{outcome.diagnostics}"
    )
