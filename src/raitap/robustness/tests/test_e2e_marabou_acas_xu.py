"""End-to-end Marabou + ACAS Xu net 1-1 verification.

Skips when:

* ``maraboupy`` is not installed (Windows, py3.13+, missing extra).
* The VNN-COMP fixture cannot be downloaded (offline runners).

Two cases:

* Tiny eps (``1e-5``) → VERIFIED. Solver returns UNSAT inside the 120s timeout.
* Huge eps (``1e6``) → FALSIFIED. Solver returns SAT with a counter-example
  whose argmax differs from the original prediction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from raitap.models.base_backend import ModelBackend

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.e2e


def _load_acas_xu_onnx() -> Path:
    """Resolve the ACAS Xu net 1-1 ONNX path; skip if unavailable."""
    try:
        from raitap.data.samples import _resolve_sample
    except Exception as error:  # pragma: no cover — import-time skip
        pytest.skip(f"raitap data sample resolver unavailable: {error}")
    try:
        cache_dir = _resolve_sample("acas_xu_n1_1")
    except Exception as error:  # network error, etc.
        pytest.skip(f"Could not download ACAS Xu fixture: {error}")
    if cache_dir is None:
        pytest.skip("ACAS Xu sample not registered.")
    onnx_files = sorted(cache_dir.glob("*.onnx"))
    if not onnx_files:
        pytest.skip(f"No ONNX file in {cache_dir}.")
    return onnx_files[0]


def _run_onnx(model_path: Path, inputs: np.ndarray) -> np.ndarray:
    """Run the ONNX model on ``inputs`` of shape ``(N, 5)`` -> ``(N, 5)``."""
    onnxruntime = pytest.importorskip("onnxruntime")
    session = onnxruntime.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    expected_shape = session.get_inputs()[0].shape
    array = np.asarray(inputs, dtype=np.float32)
    # ACAS Xu net wants (1, 1, 1, 5) — reshape generously.
    if len(expected_shape) == 4:
        array = array.reshape(array.shape[0], 1, 1, 5)
    outputs = session.run(None, {input_name: array})[0]
    return np.asarray(outputs, dtype=np.float32).reshape(array.shape[0], -1)


@pytest.fixture
def acas_xu_fixture() -> tuple[Path, torch.Tensor, int]:
    pytest.importorskip("maraboupy")
    model_path = _load_acas_xu_onnx()
    # Mid-range ACAS Xu input: rho, theta, psi, v_own, v_int.
    sample_np = np.array([[10000.0, 0.0, 0.0, 600.0, 600.0]], dtype=np.float32)
    outputs = _run_onnx(model_path, sample_np)
    target_class = int(outputs.argmax(axis=1)[0])
    return model_path, torch.from_numpy(sample_np), target_class


def test_acas_xu_tiny_eps_verified(acas_xu_fixture: tuple[Path, torch.Tensor, int]) -> None:
    from raitap.robustness.assessors import MarabouAssessor
    from raitap.robustness.contracts import (
        PerturbationBudget,
        PerturbationNorm,
        RobustnessVerdict,
    )

    model_path, sample, target_class = acas_xu_fixture

    class _OnnxBackend(ModelBackend):
        onnx_path = str(model_path)

        @property
        def hardware_label(self) -> str:
            return "test-onnx"

        def __call__(self, inputs: object, **kwargs: object) -> object:
            raise NotImplementedError("test stub: onnx-path resolution only")

    assessor = MarabouAssessor(epsilon=1e-5, timeout_s=120.0)
    assessor.check_backend_compat(_OnnxBackend())
    outcome = assessor.verify_sample(
        torch.nn.Identity(),
        sample,
        torch.tensor([target_class]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=1e-5),
        backend=_OnnxBackend(),
    )
    assert outcome.verdict == RobustnessVerdict.VERIFIED, outcome.diagnostics


def test_acas_xu_huge_eps_falsified(acas_xu_fixture: tuple[Path, torch.Tensor, int]) -> None:
    from raitap.robustness.assessors import MarabouAssessor
    from raitap.robustness.contracts import (
        PerturbationBudget,
        PerturbationNorm,
        RobustnessVerdict,
    )

    model_path, sample, target_class = acas_xu_fixture

    class _OnnxBackend(ModelBackend):
        onnx_path = str(model_path)

        @property
        def hardware_label(self) -> str:
            return "test-onnx"

        def __call__(self, inputs: object, **kwargs: object) -> object:
            raise NotImplementedError("test stub: onnx-path resolution only")

    huge_eps = 1e6  # whole input domain — adversarial example must exist.
    assessor = MarabouAssessor(epsilon=huge_eps, timeout_s=120.0)
    assessor.check_backend_compat(_OnnxBackend())
    outcome = assessor.verify_sample(
        torch.nn.Identity(),
        sample,
        torch.tensor([target_class]),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=huge_eps),
        backend=_OnnxBackend(),
    )

    assert outcome.verdict == RobustnessVerdict.FALSIFIED, outcome.diagnostics
    assert outcome.counter_example is not None
    assert tuple(outcome.counter_example.shape) == (1, 5)
    cx_outputs = _run_onnx(model_path, outcome.counter_example.detach().cpu().numpy())
    assert int(cx_outputs.argmax(axis=1)[0]) != target_class
