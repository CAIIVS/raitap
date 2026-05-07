"""Assessor base classes for the RAITAP robustness module.

Three layers, mirroring transparency's split between framework-owned and
adapter-owned pipelines:

* :class:`BaseAssessor` — root: declares ``method_kind`` and the no-op
  ``check_backend_compat`` default.
* :class:`EmpiricalAttackAssessor` — framework owns ``assess()``; subclasses
  implement only :meth:`generate_adversarial`.
* :class:`FormalVerificationAssessor` — framework owns ``assess()``; subclasses
  implement only :meth:`verify_sample` and return a per-sample
  :class:`~raitap.robustness.contracts.VerificationOutcome`.
"""

from __future__ import annotations

import gc
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from raitap.configs import resolve_run_dir

from ..contracts import (
    MethodKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessVerdict,
    ThreatModel,
    VerificationOutcome,
)
from ..results import (
    ConfiguredRobustnessVisualiser,
    RobustnessMetrics,
    RobustnessResult,
    encode_verdicts,
)
from ..semantics import assessor_semantics

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

_VISUALISATION_ONLY_KWARGS = frozenset({"sample_names", "show_sample_names"})


class BaseAssessor:
    """Root base class for all robustness assessors."""

    method_kind: ClassVar[MethodKind]
    threat_model_default: ClassVar[ThreatModel] = ThreatModel.WHITE_BOX
    objective_default: ClassVar[Objective] = Objective.UNTARGETED

    def check_backend_compat(self, backend: object) -> None:
        del backend
        return None


def _resolve_per_sample_target(
    targets: torch.Tensor,
    target_classes: tuple[int, ...] | None,
) -> torch.Tensor:
    """Return per-sample reference labels used for verdict computation."""
    if target_classes is None:
        return targets
    if len(target_classes) == 1:
        return torch.full_like(targets, fill_value=int(target_classes[0]))
    if len(target_classes) != int(targets.shape[0]):
        raise ValueError(
            "Length of target_classes must equal the batch size or be 1; "
            f"got {len(target_classes)} for batch size {int(targets.shape[0])}."
        )
    return torch.tensor(target_classes, dtype=targets.dtype, device=targets.device)


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


class EmpiricalAttackAssessor(BaseAssessor, ABC):
    """Empirical attack adapter: subclass implements one method, framework does the rest."""

    method_kind: ClassVar[MethodKind] = MethodKind.EMPIRICAL_ATTACK

    @abstractmethod
    def generate_adversarial(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        backend: object | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return a perturbed-inputs tensor matching ``inputs.shape``."""

    def assess(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        backend: object | None = None,
        run_dir: str | Path | None = None,
        output_root: str | Path = ".",
        experiment_name: str | None = None,
        assessor_target: str | None = None,
        assessor_name: str | None = None,
        visualisers: list[ConfiguredRobustnessVisualiser] | None = None,
        raitap_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RobustnessResult:
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
        clean_predictions = _argmax_predictions(model, inputs)
        adversarial_predictions = _argmax_predictions(model, perturbed)

        reference_targets = _resolve_per_sample_target(targets, semantics.target_classes)
        verdicts_list, attack_success_mask = _empirical_verdicts(
            adversarial_predictions=adversarial_predictions,
            reference_targets=reference_targets,
            objective=semantics.objective,
        )
        verdicts = encode_verdicts(verdicts_list)

        delta = (perturbed - inputs).detach().cpu()
        per_sample_distance = _per_sample_norm(delta, semantics.budget.norm)

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
            assessor_target=assessor_target or f"{type(self).__module__}.{type(self).__name__}",
            algorithm=str(getattr(self, "algorithm", "")),
            assessor_name=assessor_name,
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
    """Formal-verification adapter: subclass implements per-sample :meth:`verify_sample`."""

    method_kind: ClassVar[MethodKind] = MethodKind.FORMAL_VERIFICATION

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
        backend: object | None = None,
        run_dir: str | Path | None = None,
        output_root: str | Path = ".",
        experiment_name: str | None = None,
        assessor_target: str | None = None,
        assessor_name: str | None = None,
        visualisers: list[ConfiguredRobustnessVisualiser] | None = None,
        raitap_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RobustnessResult:
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

        clean_predictions = _argmax_predictions(model, inputs)

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
                    budget=semantics.budget,
                    backend=backend,
                    **verify_kwargs,
                )
            except Exception:  # pragma: no cover — per-sample isolation
                logger.exception("verify_sample crashed for index %d", index)
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
            norm=semantics.budget.norm,
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
            assessor_target=assessor_target or f"{type(self).__module__}.{type(self).__name__}",
            algorithm=str(getattr(self, "algorithm", "")),
            assessor_name=assessor_name,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detach_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu()


def _argmax_predictions(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else None
        if device is not None and inputs.device != device:
            outputs = model(inputs.to(device))
        else:
            outputs = model(inputs)
    if not isinstance(outputs, torch.Tensor):
        outputs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    return outputs.argmax(dim=1).detach().cpu()


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
        RobustnessVerdict.ATTACKED if hit else RobustnessVerdict.NOT_ATTACKED
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
        falsified_preds = _argmax_predictions(model, falsified_inputs)
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
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total_batches, desc=progress_desc)


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


def _normalise_optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]
