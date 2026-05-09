"""Marabou formal-verification adapter for RAITAP.

Wraps :mod:`maraboupy` (Marabou >=2.0) behind raitap's
:class:`FormalVerificationAssessor` API. Accepts both torch and ONNX backends:

* ONNX backend → re-uses ``backend.onnx_path`` directly.
* Torch backend → exports ``model`` to a temp ONNX file once per ``assess()``
  call, caches the path keyed on ``(id(model), sample.shape[1:])``.

Only static-shape MLPs are in scope. Export failures propagate as
:class:`AssessorBackendIncompatibilityError` ("fail loud").
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from ..contracts import (
    MethodKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessVerdict,
    ThreatModel,
    VerificationOutcome,
)
from ..exceptions import AssessorBackendIncompatibilityError
from ..semantics import AssessorSemanticsHints
from .base_assessor import FormalVerificationAssessor

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import nn

logger = logging.getLogger(__name__)


class MarabouAssessor(FormalVerificationAssessor):
    """Marabou-backed L∞ formal-verification adapter.

    Only ``algorithm="linf-box"`` is supported in v1: per-input box bounds
    ``[x_i - eps, x_i + eps]`` plus an output disjunction asserting "any class
    other than the target dominates the target". UNSAT → VERIFIED, SAT →
    FALSIFIED with reconstructed counter-example, TIMEOUT/UNKNOWN → UNKNOWN.

    The adapter restricts itself to static-shape MLPs (no dynamic axes,
    BatchNorm/Dropout disabled via ``model.eval()``); ONNX export failures
    are converted to :class:`AssessorBackendIncompatibilityError` so the
    user sees a clear "Marabou cannot handle this graph" message instead of
    a torch traceback.
    """

    algorithm_registry: ClassVar[Mapping[str, AssessorSemanticsHints]] = {
        "linf-box": AssessorSemanticsHints(
            MethodKind.FORMAL_VERIFICATION,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families=frozenset({"smt", "complete", "sound"}),
        ),
    }

    # Budget keys (epsilon, norm) live under ``constructor:`` in the YAML; the
    # adapter applies them at verify-time but they're configured at __init__.
    budget_kwarg_source: ClassVar[str] = "init_kwargs"

    def __init__(
        self,
        *,
        algorithm: str = "linf-box",
        timeout_s: float = 300.0,
        epsilon: float = 0.05,
        norm: str = "Linf",
        **kwargs: Any,
    ) -> None:
        del kwargs  # tolerate forward-compat kwargs for YAML configs.
        if algorithm not in type(self).algorithm_registry:
            valid = ", ".join(sorted(type(self).algorithm_registry))
            raise ValueError(f"MarabouAssessor: unknown algorithm {algorithm!r}. Known: {valid}.")
        self.algorithm = algorithm
        self.timeout_s = float(timeout_s)
        self.epsilon = float(epsilon)
        self._norm = str(norm)
        # Record kwargs the way the framework expects (matches torchattacks /
        # foolbox adapters: ``init_kwargs`` carries the budget for semantics).
        self.init_kwargs: dict[str, Any] = {
            "epsilon": float(epsilon),
            "norm": str(norm),
            "timeout_s": float(timeout_s),
        }
        # Per-(model, sample-shape) cache so we only export once per assess().
        self._onnx_cache: dict[tuple[int, tuple[int, ...]], Path] = {}
        # One-shot info log per assess() so users see "we exported" exactly once.
        self._export_logged: bool = False

    # ------------------------------------------------------------------
    # Backend acceptance
    # ------------------------------------------------------------------

    def check_backend_compat(self, backend: object) -> None:
        # Reset per-assess() state. ``check_backend_compat`` is called by
        # ``FormalVerificationAssessor.assess`` exactly once before the
        # per-sample loop, so it doubles as a per-call setup hook.
        self._onnx_cache.clear()
        self._export_logged = False
        del backend

    # ------------------------------------------------------------------
    # ONNX resolution
    # ------------------------------------------------------------------

    def _resolve_onnx_path(
        self,
        model: nn.Module,
        sample: torch.Tensor,
        backend: object | None,
    ) -> Path:
        """Return an ONNX file path that Marabou can read.

        Strategy:
        1. If ``backend`` exposes ``onnx_path`` (or ``model_path`` ending in
           ``.onnx``), reuse it directly — no export needed.
        2. Otherwise export the torch ``model`` to a temp file and cache the
           path keyed on ``(id(model), sample.shape[1:])``.
        """
        existing = _backend_onnx_path(backend)
        if existing is not None:
            return existing

        cache_key = (id(model), tuple(int(d) for d in sample.shape[1:]))
        cached = self._onnx_cache.get(cache_key)
        if cached is not None and cached.exists():
            return cached

        path = _export_torch_to_onnx(model, sample)
        self._onnx_cache[cache_key] = path
        if not self._export_logged:
            logger.info(
                "Exporting torch model to ONNX for Marabou verification "
                "(cached for subsequent samples): %s",
                path,
            )
            self._export_logged = True
        return path

    # ------------------------------------------------------------------
    # Per-sample verification
    # ------------------------------------------------------------------

    def verify_sample(
        self,
        model: nn.Module,
        sample: torch.Tensor,
        target: torch.Tensor,
        *,
        budget: PerturbationBudget,
        backend: object | None = None,
        **kwargs: Any,
    ) -> VerificationOutcome:
        del kwargs
        if budget.norm != PerturbationNorm.LINF:
            raise ValueError(
                f"MarabouAssessor only supports {PerturbationNorm.LINF.value} "
                f"perturbations; got {budget.norm.value}."
            )
        eps = float(budget.epsilon) if budget.epsilon is not None else self.epsilon

        onnx_path = self._resolve_onnx_path(model, sample, backend)

        try:
            from maraboupy import Marabou, MarabouCore  # type: ignore[import-not-found]
        except ImportError as error:
            raise ImportError(
                "MarabouAssessor requires the optional dependency 'maraboupy'. "
                "Install it with `uv sync --extra marabou` "
                "(Linux/macOS x86-64, Python 3.11-3.12 only)."
            ) from error

        network = Marabou.read_onnx(str(onnx_path))
        input_vars = np.asarray(network.inputVars[0]).reshape(-1)
        output_vars = np.asarray(network.outputVars[0]).reshape(-1)

        flat_sample = sample.detach().cpu().to(torch.float32).numpy().reshape(-1)
        if flat_sample.size != input_vars.size:
            raise AssessorBackendIncompatibilityError(
                assessor=type(self).__name__,
                backend=type(backend).__name__ if backend is not None else "?",
                algorithm=self.algorithm,
                reason=(
                    f"ONNX network expects {input_vars.size} input variables but the "
                    f"sample has {flat_sample.size} elements; check that the ONNX "
                    "graph matches the sample shape."
                ),
            )

        for var_id, value in zip(input_vars, flat_sample, strict=True):
            network.setLowerBound(int(var_id), float(value) - eps)
            network.setUpperBound(int(var_id), float(value) + eps)

        target_idx = int(target.reshape(-1)[0].item())
        if not (0 <= target_idx < output_vars.size):
            raise ValueError(
                f"target index {target_idx} out of range for output size {output_vars.size}."
            )

        disjuncts: list[list[Any]] = []
        target_var = int(output_vars[target_idx])
        for j, out_var in enumerate(output_vars):
            if j == target_idx:
                continue
            equation = MarabouCore.Equation(MarabouCore.Equation.GE)
            equation.addAddend(1.0, int(out_var))
            equation.addAddend(-1.0, target_var)
            equation.setScalar(0.0)
            disjuncts.append([equation])

        if disjuncts:
            network.addDisjunctionConstraint(disjuncts)

        options = Marabou.createOptions(timeoutInSeconds=int(self.timeout_s), verbosity=0)
        started = time.perf_counter()
        exit_code, values, stats = network.solve(options=options)
        wall_runtime = time.perf_counter() - started

        runtime_seconds = _stats_runtime_seconds(stats)
        if runtime_seconds is None:
            runtime_seconds = wall_runtime

        verdict, counter_example = _interpret_solver_result(
            exit_code=exit_code,
            values=values,
            input_vars=input_vars,
            sample_shape=tuple(int(d) for d in sample.shape),
        )

        return VerificationOutcome(
            verdict=verdict,
            counter_example=counter_example,
            lower_bounds=None,
            upper_bounds=None,
            runtime_seconds=float(runtime_seconds),
            diagnostics={"exit_code": str(exit_code)},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend_onnx_path(backend: object | None) -> Path | None:
    """Return the backend's ONNX file path if it exposes one."""
    if backend is None:
        return None
    for attr in ("onnx_path", "model_path"):
        candidate = getattr(backend, attr, None)
        if candidate is None:
            continue
        path = Path(str(candidate))
        if path.suffix.lower() == ".onnx" and path.exists():
            return path
    return None


def _export_torch_to_onnx(model: nn.Module, sample: torch.Tensor) -> Path:
    """Export a torch ``model`` to an ONNX file Marabou can read.

    Static-shape, opset 13, ``model.eval()``. Failures are wrapped in
    :class:`AssessorBackendIncompatibilityError` so the user sees a clear
    "Marabou requires static-shape MLPs" message instead of a raw torch
    traceback.
    """
    target_dir = Path(tempfile.mkdtemp(prefix="raitap-marabou-"))
    target_path = target_dir / "model.onnx"
    eval_model = model
    with contextlib.suppress(AttributeError):
        # Bare callable (e.g. mock) lacks ``.eval()`` — ignore.
        eval_model = model.eval()  # type: ignore[assignment]
    try:
        torch.onnx.export(  # type: ignore[attr-defined]
            eval_model,
            (sample.detach().to(torch.float32),),
            str(target_path),
            opset_version=13,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    except Exception as error:
        raise AssessorBackendIncompatibilityError(
            assessor="MarabouAssessor",
            backend=type(model).__name__,
            algorithm="linf-box",
            reason=(
                "Marabou requires static-shape MLPs; torch.onnx.export failed: "
                f"{error!s}. Use a torch model with fixed input shape and only "
                "Marabou-supported ops (Linear/ReLU/MaxPool/Conv variants) or "
                "supply an ONNX backend directly."
            ),
        ) from error
    return target_path


def _stats_runtime_seconds(stats: object) -> float | None:
    """Best-effort extraction of solver wall time from Marabou stats."""
    getter = getattr(stats, "getTotalTimeInMicro", None)
    if callable(getter):
        try:
            micros = getter()
        except Exception:  # pragma: no cover — defensive
            return None
        try:
            return float(micros) / 1e6  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return None


def _interpret_solver_result(
    *,
    exit_code: object,
    values: object,
    input_vars: np.ndarray,
    sample_shape: tuple[int, ...],
) -> tuple[RobustnessVerdict, torch.Tensor | None]:
    """Map Marabou's exit code + value dict to a raitap verdict + counter-example."""
    code = str(exit_code).strip().lower()
    if code in {"unsat", "valid"}:
        return RobustnessVerdict.VERIFIED, None
    if code in {"sat", "invalid"}:
        counter_example = _reconstruct_counter_example(values, input_vars, sample_shape)
        return RobustnessVerdict.FALSIFIED, counter_example
    # "TIMEOUT", "UNKNOWN", "ERROR", or any other code → UNKNOWN.
    return RobustnessVerdict.UNKNOWN, None


def _reconstruct_counter_example(
    values: object,
    input_vars: np.ndarray,
    sample_shape: tuple[int, ...],
) -> torch.Tensor | None:
    """Pull the input assignment from Marabou's ``values`` dict and reshape it."""
    if not isinstance(values, dict) or not values:
        return None
    try:
        flat = np.array(
            [float(values[int(v)]) for v in input_vars],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError):
        return None
    try:
        reshaped = flat.reshape(sample_shape)
    except ValueError:
        # Fall back to flat counter-example if reshape fails.
        reshaped = flat
    return torch.from_numpy(np.ascontiguousarray(reshaped))
