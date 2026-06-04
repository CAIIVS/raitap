from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from raitap.tracking.base_tracker import BaseTracker, Trackable
from raitap.utils.errors import SampleNamesLengthError
from raitap.utils.lazy import lazy_import
from raitap.utils.serialization import to_json_serialisable

if TYPE_CHECKING:
    import torch
else:
    torch = lazy_import("torch")

from .contracts import (
    VERDICT_CODES,
    AssessmentKind,
    RobustnessSemantics,
    RobustnessVerdict,
    RobustnessVisualisationContext,
    decode_verdict,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .visualisers.base_visualiser import BaseRobustnessVisualiser


def _serialisable(value: Any) -> Any:
    return to_json_serialisable(value)


def _serialisable_semantics(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, frozenset):
        return sorted(_serialisable_semantics(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            str(item.name): _serialisable_semantics(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _serialisable_semantics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialisable_semantics(item) for item in value]
    return _serialisable(value)


def _serialisable_call_kwarg(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(key): _serialisable_call_kwarg(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialisable_call_kwarg(item) for item in value]
    return _serialisable(value)


def _normalise_sample_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _slice_sample_tensor(tensor: torch.Tensor, sample_index: int) -> torch.Tensor:
    return tensor[sample_index : sample_index + 1]


def _slice_optional_sample_tensor(
    tensor: torch.Tensor | None,
    sample_index: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return _slice_sample_tensor(tensor, sample_index)


def _slice_output_bounds(
    output_bounds: dict[str, torch.Tensor] | None,
    *,
    batch_size: int,
    sample_index: int,
) -> dict[str, torch.Tensor] | None:
    if output_bounds is None:
        return None
    sliced: dict[str, torch.Tensor] = {}
    for key, value in output_bounds.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            sliced[key] = _slice_sample_tensor(value, sample_index)
        else:
            sliced[key] = value
    return sliced


def encode_verdicts(verdicts: list[RobustnessVerdict]) -> torch.Tensor:
    """Encode a per-sample list of verdicts as a long tensor of stable codes."""
    return torch.tensor([VERDICT_CODES[v] for v in verdicts], dtype=torch.long)


def decode_verdicts(verdicts: torch.Tensor) -> list[RobustnessVerdict]:
    return [decode_verdict(int(code)) for code in verdicts.tolist()]


@dataclass
class RobustnessMetrics:
    """Aggregate metrics for a robustness assessment.

    ``clean_accuracy`` is always populated. Empirical-only / verifier-only fields
    are populated by the matching base assessor pipeline; the unused half stays
    ``None`` and is dropped from ``as_dict``.
    """

    clean_accuracy: float
    # Empirical-only
    adversarial_accuracy: float | None = None
    attack_success_rate: float | None = None
    mean_distance: float | None = None
    max_distance: float | None = None
    # Verifier-only
    verified_rate: float | None = None
    falsified_rate: float | None = None
    unknown_rate: float | None = None
    error_rate: float | None = None
    mean_runtime: float | None = None
    # Average-case (statistical sampling) only
    corrupted_accuracy: float | None = None
    accuracy_ci_low: float | None = None
    accuracy_ci_high: float | None = None
    n_samples: int | None = None
    n_correct: int | None = None
    # Overflow
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {"clean_accuracy": float(self.clean_accuracy)}
        for name in (
            "adversarial_accuracy",
            "attack_success_rate",
            "mean_distance",
            "max_distance",
            "verified_rate",
            "falsified_rate",
            "unknown_rate",
            "error_rate",
            "mean_runtime",
            "corrupted_accuracy",
            "accuracy_ci_low",
            "accuracy_ci_high",
            "n_samples",
            "n_correct",
        ):
            value = getattr(self, name)
            if value is not None:
                out[name] = float(value)
        out.update({k: float(v) for k, v in self.metrics.items()})
        return out


@dataclass(frozen=True)
class ConfiguredRobustnessVisualiser:
    """Visualiser instance plus per-call kwargs for ``BaseRobustnessVisualiser.visualise``."""

    visualiser: BaseRobustnessVisualiser
    call_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobustnessResult(Trackable):
    """Trackable result of a robustness assessment.

    A single shape covers both empirical and formal-verification assessors.
    ``perturbed_inputs`` / ``perturbed_predictions`` / ``perturbation_distance``
    are always populated for empirical results; for formal-verification results
    they hold counter-examples from FALSIFIED rows (NaN-padded for non-FALSIFIED
    rows so the artifact stays a single tensor; mask via ``verdicts``).
    ``output_bounds`` and ``runtime_per_sample`` are verifier-only side channels.
    """

    clean_inputs: torch.Tensor
    targets: torch.Tensor
    clean_predictions: torch.Tensor
    verdicts: torch.Tensor
    metrics: RobustnessMetrics
    run_dir: Path
    experiment_name: str | None
    adapter_target: str
    algorithm: str
    name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    call_kwargs: dict[str, Any] = field(default_factory=dict)
    visualiser_targets: list[str] = field(default_factory=list)
    visualisers: list[ConfiguredRobustnessVisualiser] = field(default_factory=list, repr=False)
    # The visualisations this result owns (1:N). Populated by ``visualise()``;
    # ``compare=False``/``repr=False`` because each ``RobustnessVisualisationResult``
    # back-references this result, so the default ``__eq__``/``__repr__`` would recurse.
    visualisations: list[RobustnessVisualisationResult] = field(
        default_factory=list, repr=False, compare=False
    )
    perturbed_inputs: torch.Tensor | None = None
    perturbed_predictions: torch.Tensor | None = None
    perturbation_distance: torch.Tensor | None = None
    output_bounds: dict[str, torch.Tensor] | None = None
    runtime_per_sample: torch.Tensor | None = None
    semantics: RobustnessSemantics = field(kw_only=True)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if not isinstance(self.semantics, RobustnessSemantics):
            raise TypeError("RobustnessResult.semantics must be a RobustnessSemantics.")
        # Detach + CPU to avoid GPU memory retention.
        self.clean_inputs = self.clean_inputs.detach().cpu()
        self.targets = self.targets.detach().cpu()
        self.clean_predictions = self.clean_predictions.detach().cpu()
        self.verdicts = self.verdicts.detach().cpu().to(torch.long)
        if self.perturbed_inputs is not None:
            self.perturbed_inputs = self.perturbed_inputs.detach().cpu()
        if self.perturbed_predictions is not None:
            self.perturbed_predictions = self.perturbed_predictions.detach().cpu()
        if self.perturbation_distance is not None:
            self.perturbation_distance = self.perturbation_distance.detach().cpu()
        if self.runtime_per_sample is not None:
            self.runtime_per_sample = self.runtime_per_sample.detach().cpu()
        if self.output_bounds is not None:
            self.output_bounds = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.output_bounds.items()
            }

    @property
    def assessment_kind(self) -> AssessmentKind:
        return self.semantics.assessment_kind

    def write_artifacts(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "clean_inputs": self.clean_inputs,
            "targets": self.targets,
            "clean_predictions": self.clean_predictions,
            "verdicts": self.verdicts,
        }
        if self.perturbed_inputs is not None:
            payload["perturbed_inputs"] = self.perturbed_inputs
        if self.perturbed_predictions is not None:
            payload["perturbed_predictions"] = self.perturbed_predictions
        if self.perturbation_distance is not None:
            payload["perturbation_distance"] = self.perturbation_distance
        if self.output_bounds is not None:
            payload["output_bounds"] = self.output_bounds
        if self.runtime_per_sample is not None:
            payload["runtime_per_sample"] = self.runtime_per_sample
        torch.save(payload, self.run_dir / "robustness_data.pt")
        self._write_metadata()

    def _metadata(self, *, visualiser_targets: list[str] | None = None) -> dict[str, Any]:
        targets = self.visualiser_targets if visualiser_targets is None else visualiser_targets
        return {
            "experiment_name": self.experiment_name,
            "target": self.adapter_target,
            "algorithm": self.algorithm,
            "assessor_name": self.name,
            "assessment_kind": self.assessment_kind.value,
            "case": self.semantics.case.value,
            "visualisers": targets,
            "metrics": self.metrics.as_dict(),
            "verdict_codes": {v.value: code for v, code in VERDICT_CODES.items()},
            "semantics": _serialisable_semantics(self.semantics),
            "kwargs": {key: _serialisable(value) for key, value in self.kwargs.items()},
            "call_kwargs": {
                key: _serialisable_call_kwarg(value) for key, value in self.call_kwargs.items()
            },
        }

    def _write_metadata(self) -> None:
        (self.run_dir / "metadata.json").write_text(
            json.dumps(self._metadata(), indent=2),
            encoding="utf-8",
        )

    def _visualise(self, **kwargs: Any) -> list[RobustnessVisualisationResult]:
        results: list[RobustnessVisualisationResult] = []
        new_targets: list[str] = []

        for index, configured in enumerate(self.visualisers):
            vis = configured.visualiser
            merged_call = {**configured.call_kwargs, **kwargs}
            show_sample_names = bool(
                merged_call.pop("show_sample_names", self.kwargs.get("show_sample_names", False))
            )
            sample_names_value = merged_call.pop("sample_names", self.kwargs.get("sample_names"))
            sample_names = _normalise_sample_names(sample_names_value)

            limit = int(self.clean_inputs.shape[0])
            if sample_names and len(sample_names) != limit:
                raise SampleNamesLengthError(
                    got=len(sample_names),
                    expected=limit,
                    source="RobustnessResult._visualise",
                )

            context = RobustnessVisualisationContext(
                algorithm=self.algorithm,
                assessment_kind=self.assessment_kind,
                sample_names=sample_names,
                show_sample_names=show_sample_names,
            )

            vis.validate_result(self)
            figure = vis.visualise(self, context=context, **merged_call)

            cls = type(vis)
            visualiser_name = f"{cls.__name__}_{index}"
            output_path = self.run_dir / f"{visualiser_name}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                figure.savefig(output_path, bbox_inches="tight", dpi=150)
            finally:
                plt.close(figure)

            visualiser_target = f"{cls.__module__}.{visualiser_name}"
            if (
                visualiser_target not in self.visualiser_targets
                and visualiser_target not in new_targets
            ):
                new_targets.append(visualiser_target)

            results.append(
                RobustnessVisualisationResult(
                    result=self,
                    figure=figure,
                    visualiser_name=visualiser_name,
                    visualiser_target=visualiser_target,
                    output_path=output_path,
                )
            )

        if new_targets:
            self.visualiser_targets.extend(new_targets)
            self._write_metadata()

        # The result owns its persisted visualisations (issue #243); report +
        # tracking read them off the result instead of a parallel phase-level list.
        self.visualisations = results
        return results

    def render_visualisation_for_report(
        self,
        visualiser_index: int,
        *,
        sample_index: int | None = None,
        **render_kwargs: Any,
    ) -> RobustnessVisualisationResult | None:
        """Render one visualiser for report staging without persistence side effects.

        Unlike ``RobustnessResult.visualise``, this hook never writes PNGs,
        updates ``metadata.json``, mutates ``visualiser_targets``, or touches
        ``run_dir``. Visualiser errors propagate to the caller; callers are
        expected to pre-filter redundant visualisers before invoking this method.
        """
        batch_size = int(self.clean_inputs.shape[0])
        if sample_index is not None and not 0 <= sample_index < batch_size:
            raise IndexError(
                f"sample_index {sample_index} is out of range for batch size {batch_size}."
            )

        configured = self.visualisers[visualiser_index]
        vis = configured.visualiser
        merged_call = {**configured.call_kwargs, **render_kwargs}
        show_sample_names = bool(
            merged_call.pop("show_sample_names", self.kwargs.get("show_sample_names", False))
        )
        sample_names_value = merged_call.pop("sample_names", self.kwargs.get("sample_names"))
        sample_names = _normalise_sample_names(sample_names_value)

        if sample_names and len(sample_names) != batch_size:
            raise SampleNamesLengthError(
                got=len(sample_names),
                expected=batch_size,
                source="RobustnessResult.render",
            )

        if sample_index is None:
            result_for_render = self
        else:
            result_for_render = RobustnessResult(
                clean_inputs=_slice_sample_tensor(self.clean_inputs, sample_index),
                targets=_slice_sample_tensor(self.targets, sample_index),
                clean_predictions=_slice_sample_tensor(self.clean_predictions, sample_index),
                verdicts=_slice_sample_tensor(self.verdicts, sample_index),
                metrics=self.metrics,
                run_dir=self.run_dir,
                experiment_name=self.experiment_name,
                adapter_target=self.adapter_target,
                algorithm=self.algorithm,
                name=self.name,
                kwargs=dict(self.kwargs),
                call_kwargs=dict(self.call_kwargs),
                visualiser_targets=list(self.visualiser_targets),
                visualisers=self.visualisers,
                perturbed_inputs=_slice_optional_sample_tensor(
                    self.perturbed_inputs,
                    sample_index,
                ),
                perturbed_predictions=_slice_optional_sample_tensor(
                    self.perturbed_predictions,
                    sample_index,
                ),
                perturbation_distance=_slice_optional_sample_tensor(
                    self.perturbation_distance,
                    sample_index,
                ),
                output_bounds=_slice_output_bounds(
                    self.output_bounds,
                    batch_size=batch_size,
                    sample_index=sample_index,
                ),
                runtime_per_sample=_slice_optional_sample_tensor(
                    self.runtime_per_sample,
                    sample_index,
                ),
                semantics=self.semantics,
            )
            sample_names = [sample_names[sample_index]] if sample_names else []

        context = RobustnessVisualisationContext(
            algorithm=self.algorithm,
            assessment_kind=self.assessment_kind,
            sample_names=sample_names,
            show_sample_names=show_sample_names,
        )

        vis.validate_result(result_for_render)
        figure = vis.visualise(result_for_render, context=context, **merged_call)

        cls = type(vis)
        visualiser_name = f"{cls.__name__}_{visualiser_index}"
        visualiser_target = f"{cls.__module__}.{visualiser_name}"
        return RobustnessVisualisationResult(
            result=result_for_render,
            figure=figure,
            visualiser_name=visualiser_name,
            visualiser_target=visualiser_target,
            output_path=Path(visualiser_name).with_suffix(".png"),
        )

    def log(
        self,
        tracker: BaseTracker | None,
        artifact_path: str = "robustness",
        use_subdirectory: bool = True,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if tracker is None:
            return

        target_path = self._log_target_path(
            artifact_path=artifact_path, use_subdirectory=use_subdirectory
        )
        tracker.log_metrics(
            {f"{target_path}/{k}": v for k, v in self.metrics.as_dict().items()},
            prefix="",
        )

        if not self.visualiser_targets:
            tracker.log_artifacts(self.run_dir, target_subdirectory=target_path)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            staging_dir = Path(tmp_dir) / "robustness"
            staging_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                self.run_dir / "robustness_data.pt",
                staging_dir / "robustness_data.pt",
            )
            (staging_dir / "metadata.json").write_text(
                json.dumps(self._metadata(visualiser_targets=[]), indent=2),
                encoding="utf-8",
            )
            tracker.log_artifacts(staging_dir, target_subdirectory=target_path)

    def _log_name(self) -> str:
        return self.name or self.run_dir.name

    def _log_target_path(self, *, artifact_path: str, use_subdirectory: bool) -> str:
        assessor_name = self._log_name()
        return f"{artifact_path}/{assessor_name}" if use_subdirectory else artifact_path


@dataclass
class RobustnessVisualisationResult(Trackable):
    """PNG written to ``output_path``; ``figure`` is closed after save."""

    result: RobustnessResult
    figure: Figure
    visualiser_name: str
    visualiser_target: str
    output_path: Path

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)

    def log(
        self,
        tracker: BaseTracker | None,
        artifact_path: str = "robustness",
        use_subdirectory: bool = True,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if tracker is None:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            assessor_name = self.result._log_name()
            staging_dir = Path(tmp_dir) / assessor_name
            staging_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.output_path, staging_dir / self.output_path.name)
            target_path = self.result._log_target_path(
                artifact_path=artifact_path,
                use_subdirectory=use_subdirectory,
            )
            tracker.log_artifacts(staging_dir, target_subdirectory=target_path)
