"""Assessor base classes for the RAITAP robustness module.

Three layers, mirroring transparency's split between framework-owned and
adapter-owned pipelines:

* ``BaseAssessor`` - root: declares ``assessment_kind``. The capability gate
  (``check_backend_compat``) is inherited from ``AdapterMixin``.
* ``EmpiricalAttackAssessor`` - framework owns ``assess()`` and
  ``generate_adversarial`` (dispatcher); subclasses implement only
  ``_default_invoke``.
* ``FormalVerificationAssessor`` - framework owns ``assess()``; subclasses
  implement only ``verify_sample`` and return a per-sample
  ``VerificationOutcome``.
"""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from raitap import raitap_log
from raitap._adapters import AdapterMixin
from raitap.configs import resolve_run_dir
from raitap.models.base_backend import ModelBackend
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import numpy as np
    import torch
else:
    np = lazy_import("numpy")
    torch = lazy_import("torch")

from ..contracts import (
    AssessmentKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessVerdict,
    VerificationOutcome,
)
from ..exceptions import AssessorBackendIncompatibilityError
from ..results import (
    ConfiguredRobustnessVisualiser,
    RobustnessMetrics,
    RobustnessResult,
    encode_verdicts,
)
from ..semantics import AssessorAlgorithmSpec, assessor_semantics, hints_for_assessor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import nn

_VISUALISATION_ONLY_KWARGS = frozenset({"sample_names", "show_sample_names"})


def _require_budget(semantics: Any) -> PerturbationBudget:
    """Worst-case assessors always carry a PerturbationBudget; narrow for type-safety."""
    region = semantics.perturbation
    if not isinstance(region, PerturbationBudget):
        raise TypeError(
            f"Expected a PerturbationBudget for a worst-case assessor, got {type(region).__name__}."
        )
    return region


class BaseAssessor(AdapterMixin, ABC):
    """Root base class for all robustness assessors.

    Concrete subclasses must declare ``algorithm_registry`` as a class-body
    ClassVar — pyright errors at the decoration site if missing (the
    ``@adapters.robustness`` decorator's ``algorithm_registry`` kwarg
    is ``Required``).
    """

    algorithm_registry: ClassVar[Mapping[str, AssessorAlgorithmSpec]]
    assessment_kind: ClassVar[AssessmentKind]

    #: Which YAML block the underlying library actually consumes for budget
    #: kwargs (``eps`` / ``alpha`` / ``steps``). ``"init_kwargs"`` means the
    #: adapter forwards them at attack-instance construction (torchattacks);
    #: ``"call_kwargs"`` means they are read at attack-call time (foolbox).
    #: ``RobustnessSemantics.perturbation`` is derived from this source so reported
    #: metadata always matches what the adapter executed.
    budget_kwarg_source: ClassVar[str] = "init_kwargs"


def _resolve_per_sample_target(
    targets: torch.Tensor,
    target_classes: Sequence[int] | None,
) -> torch.Tensor:
    """Return per-sample reference labels used for verdict computation."""
    if target_classes is None:
        return targets
    target_list = list(target_classes)
    if len(target_list) == 1:
        return torch.full_like(targets, fill_value=int(target_list[0]))
    if len(target_list) != int(targets.shape[0]):
        raise ValueError(
            "Length of target_classes must equal the batch size or be 1; "
            f"got {len(target_list)} for batch size {int(targets.shape[0])}."
        )
    return torch.tensor(target_list, dtype=targets.dtype, device=targets.device)


def _per_sample_norm(delta: torch.Tensor, norm: PerturbationNorm) -> torch.Tensor:
    flat = delta.reshape(int(delta.shape[0]), -1)
    if norm == PerturbationNorm.LINF:
        return flat.abs().amax(dim=1)
    if norm == PerturbationNorm.L2:
        return flat.norm(p=2, dim=1)
    if norm == PerturbationNorm.L1:
        return flat.norm(p=1, dim=1)
    if norm == PerturbationNorm.L0:
        return (flat.abs() > 0).sum(dim=1).to(flat.dtype)
    raise ValueError(f"Unsupported perturbation norm {norm!r}.")


@dataclass(frozen=True)
class AttackInvokeCtx:
    """Per-call context handed to an empirical-attack invoker (#266).

    Carries the assessor instance so a custom invoker reuses shared helpers
    (``_rethrow``, ``_prepare_inputs_for_forward``, ``_maybe_set_targeted``,
    ``_extract_scalar_eps``, ``_build_criterion``, ``_last_success``).
    """

    assessor: EmpiricalAttackAssessor
    library: Any
    model: nn.Module
    inputs: torch.Tensor
    targets: torch.Tensor
    backend: object | None
    call_kwargs: dict[str, Any]


class EmpiricalAttackAssessor(BaseAssessor, ABC):
    """Empirical attack adapter.

    Subclasses implement only ``_default_invoke``; the framework owns the rest.
    """

    assessment_kind: ClassVar[AssessmentKind] = AssessmentKind.EMPIRICAL_ATTACK

    def generate_adversarial(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        backend: object | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Dispatch to the entry's invoker, else this adapter's default path."""
        hints = hints_for_assessor(self)
        ctx = AttackInvokeCtx(
            assessor=self,
            library=self._lazy_import(),
            model=model,
            inputs=inputs,
            targets=targets,
            backend=backend,
            call_kwargs=dict(kwargs),
        )
        invoke = hints.invoker or self._default_invoke
        return invoke(ctx)

    @abstractmethod
    def _default_invoke(self, ctx: AttackInvokeCtx) -> torch.Tensor:
        """Adapter's uniform construct-and-call path. Returns adversarials."""

    def assess(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        backend: ModelBackend | None = None,
        run_dir: str | Path | None = None,
        output_root: str | Path | None = None,
        experiment_name: str | None = None,
        assessor_target: str | None = None,
        assessor_name: str | None = None,
        visualisers: list[ConfiguredRobustnessVisualiser] | None = None,
        raitap_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RobustnessResult:
        _require_non_empty_batch(inputs, type(self).__name__)
        visualisers_list = [] if visualisers is None else visualisers
        rk = {} if raitap_kwargs is None else dict(raitap_kwargs)
        attack_kwargs = {k: v for k, v in kwargs.items() if k not in _VISUALISATION_ONLY_KWARGS}

        self.check_backend_compat(backend)

        sample_ids = _normalise_optional_str_list(rk.get("sample_ids"))
        sample_names = _normalise_optional_str_list(rk.get("sample_names"))

        semantics = assessor_semantics(
            self,
            call_kwargs=attack_kwargs,
            raitap_kwargs=rk,
            inputs=inputs,
            targets=targets,
            sample_ids=sample_ids,
            sample_names=sample_names,
        )

        batch_size = _pop_int_kwarg(rk, "batch_size")
        show_progress, progress_desc = _pop_progress_settings(rk)

        perturbed = self._compute_with_optional_batches(
            model,
            inputs,
            targets,
            attack_kwargs,
            backend,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )

        # Forward pass on clean and perturbed to derive predictions.
        clean_predictions = _argmax_predictions(model, inputs, backend=backend)
        adversarial_predictions = _argmax_predictions(model, perturbed, backend=backend)

        reference_targets = _resolve_per_sample_target(targets, semantics.target_classes)
        verdicts_list, attack_success_mask = _empirical_verdicts(
            adversarial_predictions=adversarial_predictions,
            reference_targets=reference_targets,
            objective=semantics.objective,
        )
        verdicts = encode_verdicts(verdicts_list)

        delta = (perturbed - inputs).detach().cpu()
        budget = _require_budget(semantics)
        per_sample_distance = _per_sample_norm(delta, budget.norm)

        clean_acc = (clean_predictions == targets).float().mean().item()
        adv_acc = (adversarial_predictions == targets).float().mean().item()
        attack_success_rate = (
            attack_success_mask.float().mean().item() if attack_success_mask.numel() else 0.0
        )

        metrics = RobustnessMetrics(
            clean_accuracy=float(clean_acc),
            adversarial_accuracy=float(adv_acc),
            attack_success_rate=float(attack_success_rate),
            mean_distance=float(per_sample_distance.mean().item())
            if per_sample_distance.numel()
            else 0.0,
            max_distance=float(per_sample_distance.max().item())
            if per_sample_distance.numel()
            else 0.0,
        )

        result = RobustnessResult(
            clean_inputs=inputs,
            targets=targets,
            clean_predictions=clean_predictions,
            verdicts=verdicts,
            metrics=metrics,
            run_dir=(
                Path(run_dir)
                if run_dir is not None
                else resolve_run_dir(output_root=output_root, subdir="robustness")
            ),
            experiment_name=experiment_name,
            adapter_target=assessor_target or f"{type(self).__module__}.{type(self).__name__}",
            algorithm=str(getattr(self, "algorithm", "")),
            name=assessor_name,
            kwargs={
                "sample_ids": sample_ids,
                "sample_names": sample_names,
                "show_sample_names": bool(rk.get("show_sample_names", False)),
            },
            call_kwargs=attack_kwargs,
            visualisers=visualisers_list,
            perturbed_inputs=perturbed,
            perturbed_predictions=adversarial_predictions,
            perturbation_distance=per_sample_distance,
            output_bounds=None,
            runtime_per_sample=None,
            semantics=semantics,
        )
        result.write_artifacts()
        return result

    def _compute_with_optional_batches(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        attack_kwargs: dict[str, Any],
        backend: object | None,
        *,
        batch_size: int | None,
        show_progress: bool,
        progress_desc: str | None,
    ) -> torch.Tensor:
        total_batch = int(inputs.shape[0])
        if batch_size is None or total_batch <= batch_size:
            adversarial = self.generate_adversarial(
                model, inputs, targets, backend=backend, **attack_kwargs
            )
            return _detach_cpu(adversarial)

        chunks: list[torch.Tensor] = []
        starts = range(0, total_batch, batch_size)
        if show_progress:
            starts = _wrap_with_progress(
                starts,
                total_batches=len(starts),
                progress_desc=progress_desc
                or f"{getattr(self, 'algorithm', type(self).__name__)} batches",
            )
        for start in starts:
            end = min(start + batch_size, total_batch)
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            chunk = self.generate_adversarial(
                model,
                batch_inputs,
                batch_targets,
                backend=backend,
                **attack_kwargs,
            )
            chunks.append(_detach_cpu(chunk))
            del chunk, batch_inputs, batch_targets
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(chunks, dim=0)


class FormalVerificationAssessor(BaseAssessor, ABC):
    """Formal-verification adapter: subclass implements per-sample ``verify_sample``."""

    assessment_kind: ClassVar[AssessmentKind] = AssessmentKind.FORMAL_VERIFICATION

    @abstractmethod
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
        """Verify a single sample's robustness within ``budget``."""

    def assess(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        backend: ModelBackend | None = None,
        run_dir: str | Path | None = None,
        output_root: str | Path | None = None,
        experiment_name: str | None = None,
        assessor_target: str | None = None,
        assessor_name: str | None = None,
        visualisers: list[ConfiguredRobustnessVisualiser] | None = None,
        raitap_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RobustnessResult:
        _require_non_empty_batch(inputs, type(self).__name__)
        visualisers_list = [] if visualisers is None else visualisers
        rk = {} if raitap_kwargs is None else dict(raitap_kwargs)
        verify_kwargs = {k: v for k, v in kwargs.items() if k not in _VISUALISATION_ONLY_KWARGS}

        self.check_backend_compat(backend)

        sample_ids = _normalise_optional_str_list(rk.get("sample_ids"))
        sample_names = _normalise_optional_str_list(rk.get("sample_names"))

        semantics = assessor_semantics(
            self,
            call_kwargs=verify_kwargs,
            raitap_kwargs=rk,
            inputs=inputs,
            targets=targets,
            sample_ids=sample_ids,
            sample_names=sample_names,
        )

        clean_predictions = _argmax_predictions(model, inputs, backend=backend)

        verdict_list: list[RobustnessVerdict] = []
        counter_examples: list[torch.Tensor | None] = []
        runtimes: list[float] = []
        lower_rows: list[torch.Tensor | None] = []
        upper_rows: list[torch.Tensor | None] = []

        total = int(inputs.shape[0])
        progress = _wrap_with_progress(
            range(total),
            total_batches=total,
            progress_desc=f"{getattr(self, 'algorithm', type(self).__name__)} verify",
        )
        for index in progress:
            sample = inputs[index : index + 1]
            target = targets[index : index + 1]
            started = time.perf_counter()
            try:
                outcome = self.verify_sample(
                    model,
                    sample,
                    target,
                    budget=_require_budget(semantics),
                    backend=backend,
                    **verify_kwargs,
                )
            except AssessorBackendIncompatibilityError:
                # Structural incompatibility (e.g. unsupported ONNX op) is
                # identical for every sample in the batch — propagate so the
                # user sees one clear panel instead of N repeats of the
                # generic "verify_sample crashed" log.
                raise
            except Exception as crash:  # pragma: no cover — per-sample isolation
                raitap_log.exception(
                    "verify_sample crashed for index %d: %s: %s",
                    index,
                    type(crash).__name__,
                    crash,
                )
                outcome = VerificationOutcome(
                    verdict=RobustnessVerdict.ERROR,
                    runtime_seconds=time.perf_counter() - started,
                )
            verdict_list.append(outcome.verdict)
            runtime = (
                outcome.runtime_seconds
                if outcome.runtime_seconds is not None
                else time.perf_counter() - started
            )
            runtimes.append(float(runtime))
            counter_examples.append(outcome.counter_example)
            lower_rows.append(outcome.lower_bounds)
            upper_rows.append(outcome.upper_bounds)

        verdicts = encode_verdicts(verdict_list)
        runtime_per_sample = torch.tensor(runtimes, dtype=torch.float32)

        perturbed_inputs, perturbed_predictions, perturbation_distance = _assemble_counter_examples(
            inputs=inputs,
            counter_examples=counter_examples,
            verdicts=verdict_list,
            model=model,
            norm=_require_budget(semantics).norm,
            backend=backend,
        )
        output_bounds = _stack_optional_bounds(lower_rows, upper_rows)

        clean_acc = (clean_predictions == targets).float().mean().item()
        verdict_counts = _count_verdicts(verdict_list)
        denom = max(len(verdict_list), 1)
        metrics = RobustnessMetrics(
            clean_accuracy=float(clean_acc),
            verified_rate=verdict_counts[RobustnessVerdict.VERIFIED] / denom,
            falsified_rate=verdict_counts[RobustnessVerdict.FALSIFIED] / denom,
            unknown_rate=verdict_counts[RobustnessVerdict.UNKNOWN] / denom,
            error_rate=verdict_counts[RobustnessVerdict.ERROR] / denom,
            mean_runtime=float(runtime_per_sample.mean().item()) if runtimes else 0.0,
        )

        result = RobustnessResult(
            clean_inputs=inputs,
            targets=targets,
            clean_predictions=clean_predictions,
            verdicts=verdicts,
            metrics=metrics,
            run_dir=(
                Path(run_dir)
                if run_dir is not None
                else resolve_run_dir(output_root=output_root, subdir="robustness")
            ),
            experiment_name=experiment_name,
            adapter_target=assessor_target or f"{type(self).__module__}.{type(self).__name__}",
            algorithm=str(getattr(self, "algorithm", "")),
            name=assessor_name,
            kwargs={
                "sample_ids": sample_ids,
                "sample_names": sample_names,
                "show_sample_names": bool(rk.get("show_sample_names", False)),
            },
            call_kwargs=verify_kwargs,
            visualisers=visualisers_list,
            perturbed_inputs=perturbed_inputs,
            perturbed_predictions=perturbed_predictions,
            perturbation_distance=perturbation_distance,
            output_bounds=output_bounds,
            runtime_per_sample=runtime_per_sample,
            semantics=semantics,
        )
        result.write_artifacts()
        return result


class StatisticalSamplingAssessor(BaseAssessor, ABC):
    """Average-case adapter: estimate accuracy under a perturbation distribution.

    The framework owns ``assess()``: it perturbs every image in pixel space,
    forwards the corrupted batch, scores each sample as CORRECT / MISCLASSIFIED
    against its label, and aggregates into corrupted accuracy + a binomial CI.
    Subclasses implement only ``apply_perturbation`` on a single HWC uint8 image.
    """

    assessment_kind: ClassVar[AssessmentKind] = AssessmentKind.STATISTICAL_SAMPLING

    @abstractmethod
    def apply_perturbation(self, image: np.ndarray) -> np.ndarray:
        """Corrupt one HWC uint8 [0,255] image; return the same shape and dtype."""

    def assess(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        backend: ModelBackend | None = None,
        run_dir: str | Path | None = None,
        output_root: str | Path | None = None,
        experiment_name: str | None = None,
        assessor_target: str | None = None,
        assessor_name: str | None = None,
        visualisers: list[ConfiguredRobustnessVisualiser] | None = None,
        raitap_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RobustnessResult:
        from .._ci_router import resolve_ci_settings
        from ._ci import binomial_ci

        _require_non_empty_batch(inputs, type(self).__name__)
        visualisers_list = [] if visualisers is None else visualisers
        rk = {} if raitap_kwargs is None else dict(raitap_kwargs)
        call_kwargs = {k: v for k, v in kwargs.items() if k not in _VISUALISATION_ONLY_KWARGS}

        self.check_backend_compat(backend)

        sample_ids = _normalise_optional_str_list(rk.get("sample_ids"))
        sample_names = _normalise_optional_str_list(rk.get("sample_names"))
        ci_method, ci_level = resolve_ci_settings(rk)

        semantics = assessor_semantics(
            self,
            call_kwargs=call_kwargs,
            raitap_kwargs=rk,
            inputs=inputs,
            targets=targets,
            sample_ids=sample_ids,
            sample_names=sample_names,
        )

        clean_predictions = _argmax_predictions(model, inputs, backend=backend)
        corrupted = self._corrupt_batch(inputs)
        corrupted_predictions = _argmax_predictions(model, corrupted, backend=backend)

        correct_mask = corrupted_predictions == targets
        verdict_list = [
            RobustnessVerdict.CORRECT_UNDER_PERTURBATION
            if hit
            else RobustnessVerdict.MISCLASSIFIED_UNDER_PERTURBATION
            for hit in correct_mask.tolist()
        ]
        verdicts = encode_verdicts(verdict_list)

        n_samples = int(targets.shape[0])
        n_correct = int(correct_mask.sum().item())
        accuracy = n_correct / n_samples if n_samples else 0.0
        ci_low, ci_high = binomial_ci(n_correct, n_samples, level=ci_level, method=ci_method)

        clean_acc = (clean_predictions == targets).float().mean().item()
        metrics = RobustnessMetrics(
            clean_accuracy=float(clean_acc),
            corrupted_accuracy=float(accuracy),
            accuracy_ci_low=float(ci_low),
            accuracy_ci_high=float(ci_high),
            n_samples=n_samples,
            n_correct=n_correct,
        )

        result = RobustnessResult(
            clean_inputs=inputs,
            targets=targets,
            clean_predictions=clean_predictions,
            verdicts=verdicts,
            metrics=metrics,
            run_dir=(
                Path(run_dir)
                if run_dir is not None
                else resolve_run_dir(output_root=output_root, subdir="robustness")
            ),
            experiment_name=experiment_name,
            adapter_target=assessor_target or f"{type(self).__module__}.{type(self).__name__}",
            algorithm=str(getattr(self, "algorithm", "")),
            name=assessor_name,
            kwargs={
                "sample_ids": sample_ids,
                "sample_names": sample_names,
                "show_sample_names": bool(rk.get("show_sample_names", False)),
                # Persist the resolved CI settings so saved average-case artifacts
                # are self-describing (which interval + level produced the bounds).
                "ci_method": ci_method,
                "ci_level": ci_level,
            },
            call_kwargs=call_kwargs,
            visualisers=visualisers_list,
            perturbed_inputs=corrupted,
            perturbed_predictions=corrupted_predictions,
            perturbation_distance=None,
            output_bounds=None,
            runtime_per_sample=None,
            semantics=semantics,
        )
        result.write_artifacts()
        return result

    def _corrupt_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply ``apply_perturbation`` to every image (NCHW [0,1] -> NCHW [0,1])."""
        out = torch.empty_like(inputs)
        for i in range(int(inputs.shape[0])):
            chw = inputs[i].detach().cpu().clamp(0.0, 1.0)
            hwc_uint8 = (chw.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            corrupted_hwc = np.asarray(self.apply_perturbation(hwc_uint8), dtype=np.uint8)
            corrupted_chw = torch.from_numpy(corrupted_hwc.astype(np.float32) / 255.0).permute(
                2, 0, 1
            )
            out[i] = corrupted_chw.to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detach_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu()


def _argmax_predictions(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    backend: object | None = None,
) -> torch.Tensor:
    """Forward ``inputs`` through ``model`` and return per-sample argmax predictions.

    Routes device placement through ``backend._prepare_inputs`` when a backend
    is supplied (matches the transparency module's pattern); falls back to a
    parameter-device probe for callers without a backend handle (e.g. unit
    tests using a bare ``nn.Module``).
    """
    prepared = _prepare_inputs_for_forward(inputs, model=model, backend=backend)
    with torch.no_grad():
        outputs = model(prepared)
    if not isinstance(outputs, torch.Tensor):
        outputs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    return outputs.argmax(dim=1).detach().cpu()


def _prepare_inputs_for_forward(
    inputs: torch.Tensor,
    *,
    model: nn.Module | None = None,
    backend: object | None = None,
) -> torch.Tensor:
    """Move ``inputs`` to the right device for the forward pass.

    Preference order: (1) ``backend._prepare_inputs`` (canonical entry point
    used by transparency); (2) ``next(model.parameters()).device`` fallback
    for parameter-bearing modules; (3) leave inputs untouched.
    """
    if isinstance(backend, ModelBackend):
        prepared = backend._prepare_inputs(inputs)
        if not isinstance(prepared, torch.Tensor):
            raise TypeError(
                f"backend._prepare_inputs returned {type(prepared).__name__}, expected Tensor."
            )
        return prepared
    if model is not None:
        for parameter in model.parameters():
            target = parameter.device
            return inputs if inputs.device == target else inputs.to(target)
    return inputs


def _empirical_verdicts(
    *,
    adversarial_predictions: torch.Tensor,
    reference_targets: torch.Tensor,
    objective: Objective,
) -> tuple[list[RobustnessVerdict], torch.Tensor]:
    """Return per-sample verdicts and a boolean attack-success mask."""
    if objective == Objective.TARGETED:
        success = adversarial_predictions == reference_targets
    else:
        success = adversarial_predictions != reference_targets
    verdicts = [
        RobustnessVerdict.ATTACK_SUCCEEDED if hit else RobustnessVerdict.ATTACK_FAILED
        for hit in success.tolist()
    ]
    return verdicts, success


def _count_verdicts(verdicts: list[RobustnessVerdict]) -> dict[RobustnessVerdict, int]:
    counts: dict[RobustnessVerdict, int] = dict.fromkeys(RobustnessVerdict, 0)
    for verdict in verdicts:
        counts[verdict] += 1
    return counts


def _assemble_counter_examples(
    *,
    inputs: torch.Tensor,
    counter_examples: list[torch.Tensor | None],
    verdicts: list[RobustnessVerdict],
    model: nn.Module,
    norm: PerturbationNorm,
    backend: object | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if not any(ce is not None for ce in counter_examples):
        return None, None, None

    perturbed = inputs.clone().to(torch.float32)
    perturbed.fill_(float("nan"))
    distance = torch.full((int(inputs.shape[0]),), float("nan"), dtype=torch.float32)
    for i, ce in enumerate(counter_examples):
        if ce is None:
            continue
        sample = ce.detach().cpu()
        if sample.shape != inputs[i : i + 1].shape and sample.shape == inputs[i].shape:
            sample = sample.unsqueeze(0)
        perturbed[i : i + 1] = sample.to(perturbed.dtype)
        delta = (sample - inputs[i : i + 1]).to(perturbed.dtype)
        distance[i] = _per_sample_norm(delta, norm)[0]

    # Predictions only for FALSIFIED rows; fill non-FALSIFIED with -1 sentinel.
    perturbed_predictions = torch.full((int(inputs.shape[0]),), fill_value=-1, dtype=torch.long)
    falsified_indices = [i for i, v in enumerate(verdicts) if v == RobustnessVerdict.FALSIFIED]
    if falsified_indices:
        falsified_inputs = perturbed[falsified_indices].nan_to_num(nan=0.0)
        falsified_preds = _argmax_predictions(model, falsified_inputs, backend=backend)
        for slot, value in zip(falsified_indices, falsified_preds.tolist(), strict=False):
            perturbed_predictions[slot] = int(value)
    return perturbed, perturbed_predictions, distance


def _stack_optional_bounds(
    lower_rows: list[torch.Tensor | None],
    upper_rows: list[torch.Tensor | None],
) -> dict[str, torch.Tensor] | None:
    if not any(row is not None for row in lower_rows) and not any(
        row is not None for row in upper_rows
    ):
        return None
    width = next(
        (row.numel() for row in lower_rows + upper_rows if row is not None),
        0,
    )
    if width == 0:
        return None

    def _pad(rows: list[torch.Tensor | None]) -> torch.Tensor:
        out = torch.full((len(rows), width), float("nan"), dtype=torch.float32)
        for i, row in enumerate(rows):
            if row is None:
                continue
            out[i] = row.detach().cpu().to(torch.float32).reshape(-1)[:width]
        return out

    return {"lower": _pad(lower_rows), "upper": _pad(upper_rows)}


def _wrap_with_progress(
    iterable: range,
    *,
    total_batches: int,
    progress_desc: str,
) -> Any:
    from raitap.utils.console import iter_with_progress

    return iter_with_progress(iterable, total=total_batches, desc=progress_desc)


def _pop_int_kwarg(raitap_kwargs: dict[str, Any], key: str) -> int | None:
    value = raitap_kwargs.pop(key, None)
    if value is None:
        return None
    if not isinstance(value, int):
        raise TypeError(f"raitap.{key} must be an int, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"raitap.{key} must be > 0, got {value}.")
    return value


def _pop_progress_settings(raitap_kwargs: dict[str, Any]) -> tuple[bool, str | None]:
    show_progress = raitap_kwargs.pop("show_progress", True)
    if not isinstance(show_progress, bool):
        raise TypeError(f"raitap.show_progress must be a bool, got {type(show_progress).__name__}.")
    progress_desc = raitap_kwargs.pop("progress_desc", None)
    if progress_desc is not None and not isinstance(progress_desc, str):
        raise TypeError(f"raitap.progress_desc must be a str, got {type(progress_desc).__name__}.")
    return show_progress, progress_desc


def _require_non_empty_batch(inputs: torch.Tensor, assessor_name: str) -> None:
    """Refuse to assess an empty batch.

    ``subplots(0, 3)`` crashes, ``tensor.max()`` on an empty tensor errors, and
    metrics like ``attack_success_rate`` are undefined on N=0. Bail loudly with
    context instead of producing a confusing downstream traceback.
    """
    if inputs.ndim == 0 or int(inputs.shape[0]) == 0:
        raise ValueError(
            f"{assessor_name}.assess() received an empty batch (shape={tuple(inputs.shape)}); "
            "configure the data source so at least one sample is available."
        )


def _normalise_optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]
