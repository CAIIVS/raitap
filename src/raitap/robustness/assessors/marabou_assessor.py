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
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from raitap.utils.diagnostics import Subsystem
from raitap.utils.errors import rethrow

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

# Curated error patterns for confusing maraboupy errors. Matched against
# ``str(exc)``; first hit wins. See :func:`raitap.utils.errors.rethrow`.
_MARABOUPY_ERROR_MESSAGES: Mapping[re.Pattern[str], str] = {
    re.compile(r"Invoked with: <class 'maraboupy\.MarabouCore\.Equation\.EquationType'>"): (
        "maraboupy 2.0 has a bug in `InputQueryBuilder.getInputQuery` that "
        "trips whenever a query carries equations submitted via "
        "`addDisjunctionConstraint` (it passes the `EquationType` class to the "
        "`Equation` constructor instead of an enum member). The raitap "
        "adapter avoids this path by issuing one solve per non-target class, "
        "but if you see this error a code path still routes through the "
        "broken builder — file an issue. Upstream tracker: "
        "https://github.com/NeuralNetworkVerification/Marabou/issues"
    ),
    re.compile(r"Operation \S+ not implemented"): (
        "Marabou's ONNX parser does not implement an op used by the model. "
        "Re-export with the Marabou-supported subset only — see "
        "https://github.com/NeuralNetworkVerification/Marabou/blob/master/"
        "maraboupy/parsers/ONNXParser.py for the authoritative list."
    ),
}

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

        flat_sample = sample.detach().cpu().to(torch.float32).numpy().reshape(-1)
        sample_shape = tuple(int(d) for d in sample.shape)

        with rethrow(
            subsystem=Subsystem.robustness,
            third_party_lib="maraboupy",
            message_map=_MARABOUPY_ERROR_MESSAGES,
        ):
            verdict, counter_example, runtime_seconds, exit_code, _input_vars, output_vars = (
                _verify_via_per_class_solves(
                    marabou=Marabou,
                    marabou_core=MarabouCore,
                    onnx_path=onnx_path,
                    flat_sample=flat_sample,
                    sample_shape=sample_shape,
                    target_idx_raw=int(target.reshape(-1)[0].item()),
                    eps=eps,
                    total_timeout_s=self.timeout_s,
                    assessor_name=type(self).__name__,
                    backend_name=type(backend).__name__ if backend is not None else "?",
                    algorithm=self.algorithm,
                )
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


def _verify_via_per_class_solves(
    *,
    marabou: Any,
    marabou_core: Any,
    onnx_path: Path,
    flat_sample: np.ndarray,
    sample_shape: tuple[int, ...],
    target_idx_raw: int,
    eps: float,
    total_timeout_s: float,
    assessor_name: str,
    backend_name: str,
    algorithm: str,
) -> tuple[
    RobustnessVerdict,
    torch.Tensor | None,
    float,
    object,
    np.ndarray,
    np.ndarray,
]:
    """Issue one Marabou solve per non-target class and aggregate the result.

    Replaces the natural ``addDisjunctionConstraint`` approach because
    maraboupy 2.0's ``InputQueryBuilder.getInputQuery`` crashes whenever the
    query carries equations added that way (it does
    ``MarabouCore.Equation(e.EquationType)`` — the class attribute lookup
    returns the *class* not an enum member, which the constructor rejects).

    Returns ``(verdict, counter_example, runtime_seconds, exit_code,
    input_vars, output_vars)``. Stops early on the first SAT (falsified).
    Aggregates: any SAT → FALSIFIED; all UNSAT → VERIFIED; otherwise UNKNOWN.
    """
    network = marabou.read_onnx(str(onnx_path))
    input_vars = np.asarray(network.inputVars[0]).reshape(-1)
    output_vars = np.asarray(network.outputVars[0]).reshape(-1)

    if flat_sample.size != input_vars.size:
        raise AssessorBackendIncompatibilityError(
            assessor=assessor_name,
            backend=backend_name,
            algorithm=algorithm,
            reason=(
                f"ONNX network expects {input_vars.size} input variables but "
                f"the sample has {flat_sample.size} elements; check that the "
                "ONNX graph matches the sample shape."
            ),
        )

    if not (0 <= target_idx_raw < output_vars.size):
        raise ValueError(
            f"target index {target_idx_raw} out of range for output size {output_vars.size}."
        )

    non_target = [j for j in range(output_vars.size) if j != target_idx_raw]
    per_call_timeout = max(1, math.ceil(total_timeout_s / max(1, len(non_target))))
    options = marabou.createOptions(timeoutInSeconds=per_call_timeout, verbosity=0)

    saw_unknown = False
    last_exit_code: object = "unsat"
    aggregated_runtime = 0.0
    wall_started = time.perf_counter()
    for j in non_target:
        net = marabou.read_onnx(str(onnx_path))
        ivars = np.asarray(net.inputVars[0]).reshape(-1)
        ovars = np.asarray(net.outputVars[0]).reshape(-1)
        for var_id, value in zip(ivars, flat_sample, strict=True):
            net.setLowerBound(int(var_id), float(value) - eps)
            net.setUpperBound(int(var_id), float(value) + eps)
        eq = marabou_core.Equation(marabou_core.Equation.GE)
        eq.addAddend(1.0, int(ovars[j]))
        eq.addAddend(-1.0, int(ovars[target_idx_raw]))
        eq.setScalar(0.0)
        net.addEquation(eq)
        per_call_started = time.perf_counter()
        exit_code, values, stats = net.solve(options=options)
        per_call_wall = time.perf_counter() - per_call_started
        stats_seconds = _stats_runtime_seconds(stats)
        aggregated_runtime += stats_seconds if stats_seconds is not None else per_call_wall
        last_exit_code = exit_code
        code = str(exit_code).strip().lower()
        if code in {"sat", "invalid"}:
            verdict, counter_example = _interpret_solver_result(
                exit_code=exit_code,
                values=values,
                input_vars=input_vars,
                sample_shape=sample_shape,
            )
            return (
                verdict,
                counter_example,
                aggregated_runtime,
                exit_code,
                input_vars,
                output_vars,
            )
        if code in {"error", "quit_requested"}:
            return (
                RobustnessVerdict.ERROR,
                None,
                aggregated_runtime,
                exit_code,
                input_vars,
                output_vars,
            )
        if code not in {"unsat", "valid"}:
            saw_unknown = True

    # Fallback wall clock when no per-call stats were available.
    if aggregated_runtime == 0.0:
        aggregated_runtime = time.perf_counter() - wall_started
    if saw_unknown:
        return (
            RobustnessVerdict.UNKNOWN,
            None,
            aggregated_runtime,
            last_exit_code,
            input_vars,
            output_vars,
        )
    return (
        RobustnessVerdict.VERIFIED,
        None,
        aggregated_runtime,
        last_exit_code,
        input_vars,
        output_vars,
    )


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
    options = Marabou.createOptions(timeoutInSeconds=max(1, math.ceil(timeout_s)), verbosity=0)
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
