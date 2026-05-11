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

import logging
import math
import shutil
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
        compute_output_bounds: bool = False,
        bound_search_range: float = 1e3,
        bound_tolerance: float = 1e-2,
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
        self.compute_output_bounds = bool(compute_output_bounds)
        self.bound_search_range = float(bound_search_range)
        self.bound_tolerance = float(bound_tolerance)
        if self.bound_tolerance <= 0:
            raise ValueError("MarabouAssessor: bound_tolerance must be > 0.")
        if self.bound_search_range <= 0:
            raise ValueError("MarabouAssessor: bound_search_range must be > 0.")
        # Record kwargs the way the framework expects (matches torchattacks /
        # foolbox adapters: ``init_kwargs`` carries the budget for semantics).
        self.init_kwargs: dict[str, Any] = {
            "epsilon": float(epsilon),
            "norm": str(norm),
            "timeout_s": float(timeout_s),
            "compute_output_bounds": bool(compute_output_bounds),
            "bound_search_range": float(bound_search_range),
            "bound_tolerance": float(bound_tolerance),
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
        # per-sample loop, so it doubles as a per-call setup hook. Any temp
        # directories created by the previous run get torn down here so we
        # don't leak across repeated ``assess()`` invocations on long-lived
        # assessor instances.
        self._cleanup_export_temp_dirs()
        self._onnx_cache.clear()
        self._export_logged = False
        del backend

    def _cleanup_export_temp_dirs(self) -> None:
        """Remove temp directories created by previous torch-export passes."""
        for path in list(self._onnx_cache.values()):
            target_dir = path.parent
            if str(target_dir.name).startswith("raitap-marabou-"):
                shutil.rmtree(target_dir, ignore_errors=True)

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
                "(Linux/macOS x86-64, Python 3.11 only — maraboupy 2.0 ships cp311 wheels only)."
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

        options = Marabou.createOptions(
            timeoutInSeconds=max(1, int(math.ceil(self.timeout_s))), verbosity=0
        )
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

        lower_bounds: torch.Tensor | None = None
        upper_bounds: torch.Tensor | None = None
        if self.compute_output_bounds and verdict == RobustnessVerdict.VERIFIED:
            lower_bounds, upper_bounds = _compute_output_bounds(
                onnx_path=onnx_path,
                flat_sample=flat_sample,
                eps=eps,
                num_outputs=int(output_vars.size),
                search_range=self.bound_search_range,
                tolerance=self.bound_tolerance,
                timeout_s=self.timeout_s,
            )

        return VerificationOutcome(
            verdict=verdict,
            counter_example=counter_example,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
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
    try:
        # Bare callable (e.g. mock) lacks ``.eval()`` — fall back to model.
        eval_model = model.eval()  # type: ignore[assignment]
    except AttributeError:
        eval_model = model
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
    if code in {"error", "quit_requested"}:
        # Solver-level failure: distinct from "ran out of time" (TIMEOUT) so
        # downstream metrics (error_rate) can separate them.
        return RobustnessVerdict.ERROR, None
    # "timeout", "unknown", or any other code → UNKNOWN.
    return RobustnessVerdict.UNKNOWN, None


def _bisect_output_bound(
    *,
    onnx_path: Path,
    flat_sample: np.ndarray,
    eps: float,
    output_index: int,
    mode: str,
    search_range: float,
    tolerance: float,
    timeout_s: float,
) -> float:
    """Return certified per-logit bound via bisection-via-SAT.

    ``mode='lower'``: largest c such that out[k] >= c is provable.
    ``mode='upper'``: smallest c such that out[k] <= c is provable.

    Per spec, a fresh ``MarabouNetwork`` is built for every probe.
    """
    if mode not in {"lower", "upper"}:
        raise ValueError(f"_bisect_output_bound: mode must be 'lower'|'upper', got {mode!r}")
    if search_range <= 0 or tolerance <= 0:
        raise ValueError(
            f"_bisect_output_bound: search_range and tolerance must be > 0, "
            f"got search_range={search_range}, tolerance={tolerance}"
        )

    from maraboupy import Marabou  # type: ignore[import-not-found]

    lo, hi = -float(search_range), float(search_range)
    options = Marabou.createOptions(
        timeoutInSeconds=max(1, int(math.ceil(timeout_s))), verbosity=0
    )
    max_iters = max(1, math.ceil(math.log2((2.0 * search_range) / tolerance)) + 2)
    had_certifying_unsat = False
    for _ in range(max_iters):
        if (hi - lo) <= tolerance:
            break
        mid = (lo + hi) / 2.0
        network = Marabou.read_onnx(str(onnx_path))
        input_vars = np.asarray(network.inputVars[0]).reshape(-1)
        output_vars = np.asarray(network.outputVars[0]).reshape(-1)
        for var_id, value in zip(input_vars, flat_sample, strict=True):
            network.setLowerBound(int(var_id), float(value) - eps)
            network.setUpperBound(int(var_id), float(value) + eps)
        out_var = int(output_vars[output_index])
        if mode == "lower":
            network.setUpperBound(out_var, mid)
        else:
            network.setLowerBound(out_var, mid)
        exit_code, _, _ = network.solve(options=options)
        code = str(exit_code).strip().lower()
        if code in {"unsat", "valid"}:
            decision = "unsat"
            had_certifying_unsat = True
        elif code in {"sat", "invalid"}:
            decision = "sat"
        else:
            # TIMEOUT / UNKNOWN / ERROR: can't conclude. Stop bisection; current
            # conservative endpoint (lo for mode=lower, hi for mode=upper) is
            # still valid — just looser than the true bound.
            break
        if mode == "lower":
            if decision == "unsat":
                lo = mid
            else:
                hi = mid
        else:
            if decision == "unsat":
                hi = mid
            else:
                lo = mid
    if not had_certifying_unsat:
        logger.warning(
            "_bisect_output_bound: no certifying UNSAT probe for output_index=%d "
            "(mode=%s) — returning NaN. Increase bound_search_range or timeout_s.",
            output_index,
            mode,
        )
        return float("nan")
    return lo if mode == "lower" else hi


def _compute_output_bounds(
    *,
    onnx_path: Path,
    flat_sample: np.ndarray,
    eps: float,
    num_outputs: int,
    search_range: float,
    tolerance: float,
    timeout_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(lower, upper)`` tensors of shape ``(num_outputs,)``."""
    lower = np.empty(num_outputs, dtype=np.float32)
    upper = np.empty(num_outputs, dtype=np.float32)
    for k in range(num_outputs):
        lower[k] = _bisect_output_bound(
            onnx_path=onnx_path,
            flat_sample=flat_sample,
            eps=eps,
            output_index=k,
            mode="lower",
            search_range=search_range,
            tolerance=tolerance,
            timeout_s=timeout_s,
        )
        upper[k] = _bisect_output_bound(
            onnx_path=onnx_path,
            flat_sample=flat_sample,
            eps=eps,
            output_index=k,
            mode="upper",
            search_range=search_range,
            tolerance=tolerance,
            timeout_s=timeout_s,
        )
    return (
        torch.from_numpy(np.ascontiguousarray(lower)),
        torch.from_numpy(np.ascontiguousarray(upper)),
    )


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
