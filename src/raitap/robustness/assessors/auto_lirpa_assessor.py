"""auto-LiRPA formal-verification adapter for RAITAP.

Wraps `auto_LiRPA <https://github.com/Verified-Intelligence/auto_LiRPA>`_ — a
*sound but incomplete* robustness verifier that propagates certified output
bounds (CROWN / IBP) directly over a torch model. Complements the Marabou
adapter (complete SMT, static-shape MLPs only) with a bound-propagation method
that scales to CNNs and supports L∞ / L2 without an ONNX export step.

Contract mapping onto :class:`FormalVerificationAssessor`:

* ``compute_bounds()`` returns ``(lower, upper)`` per-class logit bounds →
  populate ``VerificationOutcome.lower_bounds`` / ``upper_bounds`` for **every**
  sample (the bounds *are* the certificate, computed whether or not the verdict
  is VERIFIED — unlike Marabou, which only fills them on VERIFIED).
* Verdict: ``lb[true] > max(ub[other classes])`` → ``VERIFIED``, else
  ``UNKNOWN``. Sound + incomplete → never ``FALSIFIED``, no counter-example.
* Torch-only: needs autograd and the live ``nn.Module``; ONNX backends are
  rejected by the inherited :meth:`BaseAssessor.check_backend_compat`.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

from raitap.robustness.assessors.registration import robustness_adapter
from raitap.utils.lazy import lazy_import

from ..contracts import (
    AssessmentKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessVerdict,
    ThreatModel,
    VerificationOutcome,
)
from ..semantics import AssessorSemanticsHints
from .base_assessor import FormalVerificationAssessor

if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    torch = lazy_import("torch")

# Algorithm key → (auto-LiRPA ``method`` string, perturbation norm). The key is
# the single source of truth for both the bound-propagation method and the norm
# (the norm hint is fixed per algorithm, so a separate ``method``/``norm`` kwarg
# would be a redundant second source). IBP is interval/L∞ only — it is not
# paired with L2.
_ALGORITHMS: dict[str, tuple[str, PerturbationNorm]] = {
    "crown": ("CROWN", PerturbationNorm.LINF),
    "ibp": ("IBP", PerturbationNorm.LINF),
    "crown-ibp": ("CROWN-IBP", PerturbationNorm.LINF),
    "crown-l2": ("CROWN", PerturbationNorm.L2),
}

_FAMILIES = frozenset({"bound-propagation", "incomplete", "sound"})

# Perturbation norm → auto-LiRPA ``PerturbationLpNorm(norm=...)`` value.
_NORM_TO_LIRPA: dict[PerturbationNorm, float] = {
    PerturbationNorm.LINF: float("inf"),
    PerturbationNorm.L2: 2.0,
    PerturbationNorm.L1: 1.0,
}


@robustness_adapter(
    registry_name="auto_lirpa",
    extra="auto-lirpa",
    library="auto_LiRPA",
    algorithm_registry={
        name: AssessorSemanticsHints(
            AssessmentKind.FORMAL_VERIFICATION,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            norm,
            families=_FAMILIES,
        )
        for name, (_method, norm) in _ALGORITHMS.items()
    },
)
class AutoLiRPAAssessor(FormalVerificationAssessor):
    """auto-LiRPA-backed certified-robustness adapter (bound propagation).

    ``algorithm`` selects both the bound-propagation method and the perturbation
    norm: ``"crown"`` / ``"ibp"`` / ``"crown-ibp"`` (all L∞) and ``"crown-l2"``
    (L2). Each VERIFIED / UNKNOWN sample carries per-class certified logit
    bounds in ``VerificationOutcome``.
    """

    def __init__(
        self,
        *,
        algorithm: str = "crown",
        epsilon: float = 0.05,
        **kwargs: Any,
    ) -> None:
        del kwargs  # tolerate forward-compat kwargs for YAML configs.
        if algorithm not in _ALGORITHMS:
            valid = ", ".join(sorted(_ALGORITHMS))
            raise ValueError(f"AutoLiRPAAssessor: unknown algorithm {algorithm!r}. Known: {valid}.")
        self.algorithm = algorithm
        self.epsilon = float(epsilon)
        # ``budget_kwarg_source`` defaults to ``"init_kwargs"`` — semantics reads
        # the budget epsilon from here.
        self.init_kwargs: dict[str, Any] = {"epsilon": float(epsilon)}

    # ------------------------------------------------------------------
    # Backend acceptance
    # ------------------------------------------------------------------

    def check_backend_compat(self, backend: object) -> None:
        """Enforce the torch-autograd contract, then warn on Intel XPU.

        Extends (does not replace) the base autograd enforcement — auto-LiRPA
        needs the live ``nn.Module`` and autograd, so ONNX backends are
        rejected. auto-LiRPA has no upstream XPU support; it runs on the active
        device but less-common ops (Patches-mode conv bounds) may hit XPU op
        gaps. Warn once per ``assess()`` rather than erroring so XPU users can
        fall back to a CPU backend themselves.
        """
        super().check_backend_compat(backend)
        device = getattr(backend, "device", None)
        if getattr(device, "type", None) == "xpu":
            from raitap import raitap_log

            raitap_log.warn("auto-LiRPA is not guaranteed to support Intel GPUs.")

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
        del backend, kwargs
        method, _norm = _ALGORITHMS[self.algorithm]
        norm_value = _norm_to_lirpa(budget.norm)
        eps = float(budget.epsilon) if budget.epsilon is not None else self.epsilon

        device = _model_device(model)
        sample = sample.detach().to(device)
        with contextlib.suppress(AttributeError):
            model.eval()  # disable BatchNorm/Dropout; bare callable (mock) has no .eval().

        bounded_module_cls, bounded_tensor_cls, perturbation_cls = self._lirpa_classes()

        started = time.perf_counter()
        with self._rethrow():
            bounded_model = bounded_module_cls(model, sample)
            perturbation = perturbation_cls(norm=norm_value, eps=eps)
            bounded_input = bounded_tensor_cls(sample, perturbation)
            lower, upper = bounded_model.compute_bounds(x=(bounded_input,), method=method)
        runtime_seconds = time.perf_counter() - started

        lower_bounds = lower[0].detach().cpu().to(torch.float32)
        upper_bounds = upper[0].detach().cpu().to(torch.float32)
        num_classes = int(lower_bounds.numel())

        target_idx = int(target.reshape(-1)[0].item())
        if not (0 <= target_idx < num_classes):
            raise ValueError(
                f"target index {target_idx} out of range for output size {num_classes}."
            )

        verdict = _verdict_from_bounds(lower_bounds, upper_bounds, target_idx)
        return VerificationOutcome(
            verdict=verdict,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            runtime_seconds=float(runtime_seconds),
            diagnostics={"method": method, "norm": budget.norm.value},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lirpa_classes(self) -> tuple[type, type, type]:
        """Resolve ``(BoundedModule, BoundedTensor, PerturbationLpNorm)`` lazily.

        Raises a clear install-hint-bearing :class:`ImportError` (via
        :meth:`AdapterMixin._lazy_import`) when auto-LiRPA is missing.
        """
        auto_lirpa = self._lazy_import()
        return (
            auto_lirpa.BoundedModule,
            auto_lirpa.BoundedTensor,
            auto_lirpa.PerturbationLpNorm,
        )


def _norm_to_lirpa(norm: PerturbationNorm) -> float:
    try:
        return _NORM_TO_LIRPA[norm]
    except KeyError as error:
        valid = ", ".join(sorted(n.value for n in _NORM_TO_LIRPA))
        raise ValueError(
            f"AutoLiRPAAssessor does not support {norm.value} perturbations; "
            f"supported norms: {valid}."
        ) from error


def _model_device(model: nn.Module) -> torch.device:
    """Return the model's parameter device, falling back to CPU.

    Catches ``AttributeError`` for bare callables (mocks) without ``parameters``
    and ``StopIteration`` for parameter-less modules.
    """
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _verdict_from_bounds(
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
    target_idx: int,
) -> RobustnessVerdict:
    """``VERIFIED`` iff the target's certified lower bound dominates every other
    class's certified upper bound; ``UNKNOWN`` otherwise (never ``FALSIFIED`` —
    the verifier is sound but incomplete)."""
    target_lower = float(lower_bounds[target_idx].item())
    other_uppers = upper_bounds.clone()
    other_uppers[target_idx] = float("-inf")  # exclude the target class itself.
    max_other_upper = float(other_uppers.max().item())
    if target_lower > max_other_upper:
        return RobustnessVerdict.VERIFIED
    return RobustnessVerdict.UNKNOWN
