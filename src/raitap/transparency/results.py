from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import torch

from raitap.reporting.sections import Reportable, ReportGroup
from raitap.tracking.base_tracker import BaseTracker, Trackable
from raitap.utils.serialization import to_json_serialisable

from .contracts import ExplanationPayloadKind, VisualisationContext

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .visualisers import BaseVisualiser


def _serialisable(value: Any) -> Any:
    return to_json_serialisable(value)


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


def _batch_size(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        if len(shape) == 0:
            return None
        return int(shape[0])
    except (TypeError, ValueError):
        return None


def _normalise_sample_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _sample_names_title(sample_names: list[str]) -> str:
    first = sample_names[0]
    remaining = len(sample_names) - 1
    return first if remaining <= 0 else f"{first} (+{remaining})"


def _report_scope_for_visualiser(visualiser: BaseVisualiser) -> str:
    raw = getattr(type(visualiser), "report_scope", "local")
    return "global" if str(raw).strip().lower() == "global" else "local"


@dataclass(frozen=True)
class ConfiguredVisualiser:
    """Visualiser instance plus per-call kwargs for ``BaseVisualiser.visualise``."""

    visualiser: BaseVisualiser
    call_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationResult(Trackable, Reportable):
    attributions: torch.Tensor
    inputs: torch.Tensor
    run_dir: Path
    experiment_name: str | None
    explainer_target: str
    algorithm: str
    explainer_name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    call_kwargs: dict[str, Any] = field(default_factory=dict)
    visualiser_targets: list[str] = field(default_factory=list)
    visualisers: list[ConfiguredVisualiser] = field(default_factory=list, repr=False)
    payload_kind: ExplanationPayloadKind = ExplanationPayloadKind.ATTRIBUTIONS

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        # Ensure tensors are detached and on CPU to avoid GPU memory retention
        self.attributions = self.attributions.detach().cpu()
        self.inputs = self.inputs.detach().cpu()

    def to_report_group(self) -> ReportGroup:
        return ReportGroup(
            heading=f"Explainer: {self.explainer_name or self.algorithm}",
            images=tuple(sorted(self.run_dir.glob("*.png"))),
        )

    def write_artifacts(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.payload_kind == ExplanationPayloadKind.ATTRIBUTIONS:
            torch.save(self.attributions, self.run_dir / "attributions.pt")
        elif self.payload_kind == ExplanationPayloadKind.STRUCTURED:
            raise NotImplementedError(
                "Persistence for ExplanationPayloadKind.STRUCTURED is not implemented yet."
            )
        else:
            raise NotImplementedError(
                f"Persistence for payload kind {self.payload_kind!r} is not implemented yet."
            )
        self._write_metadata()

    def _metadata(self, *, visualiser_targets: list[str] | None = None) -> dict[str, Any]:
        targets = self.visualiser_targets if visualiser_targets is None else visualiser_targets
        return {
            "experiment_name": self.experiment_name,
            "target": self.explainer_target,
            "algorithm": self.algorithm,
            "visualisers": targets,
            "payload_kind": self.payload_kind.value,
            "kwargs": {key: _serialisable(value) for key, value in self.kwargs.items()},
            "call_kwargs": {
                key: _serialisable_call_kwarg(value) for key, value in self.call_kwargs.items()
            },
        }

    def _write_metadata(self) -> None:
        metadata_path = self.run_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(self._metadata(), indent=2),
            encoding="utf-8",
        )

    def visualise(self, **kwargs: Any) -> list[VisualisationResult]:
        results: list[VisualisationResult] = []
        new_targets: list[str] = []

        for index, configured in enumerate(self.visualisers):
            vis = configured.visualiser
            merged_call = {**configured.call_kwargs, **kwargs}
            attributions = merged_call.pop("attributions", self.attributions)
            inputs = merged_call.pop("inputs", self.inputs)
            show_sample_names = bool(
                merged_call.pop("show_sample_names", self.kwargs.get("show_sample_names", False))
            )
            sample_names_value = merged_call.pop("sample_names", self.kwargs.get("sample_names"))
            sample_names = _normalise_sample_names(sample_names_value)

            limit = _batch_size(attributions) or _batch_size(inputs)
            if limit is not None:
                sample_names = sample_names[:limit]

            # Standard RAITAP pipeline metadata
            context = VisualisationContext(
                algorithm=self.algorithm,
                sample_names=sample_names,
                show_sample_names=show_sample_names,
            )

            # Standard visualise() call with context and library-specific kwargs
            figure = vis.visualise(attributions, inputs=inputs, context=context, **merged_call)

            # Legacy fallback for visualisers that don't handle titles themselves via context
            if (
                show_sample_names
                and sample_names
                and not figure.texts
                and not any(ax.get_title() for ax in figure.axes)
            ):
                figure.suptitle(_sample_names_title(sample_names), fontsize=10)
                figure.tight_layout()

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
                VisualisationResult(
                    explanation=self,
                    figure=figure,
                    visualiser_name=visualiser_name,
                    visualiser_target=visualiser_target,
                    output_path=output_path,
                    report_scope=_report_scope_for_visualiser(vis),
                )
            )

        if new_targets:
            self.visualiser_targets.extend(new_targets)
            self._write_metadata()

        return results

    def has_visualisations_for_scope(self, scope: str) -> bool:
        wanted = "global" if scope == "global" else "local"
        return any(
            _report_scope_for_visualiser(configured.visualiser) == wanted
            for configured in self.visualisers
        )

    def save_visualisations_for_report(
        self,
        output_dir: Path,
        *,
        scope: str,
        file_stem_prefix: str,
        sample_index: int | None = None,
    ) -> tuple[Path, ...]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        selected_paths: list[Path] = []
        wanted = "global" if scope == "global" else "local"

        for index, configured in enumerate(self.visualisers):
            vis = configured.visualiser
            if _report_scope_for_visualiser(vis) != wanted:
                continue

            merged_call = dict(configured.call_kwargs)
            attributions = merged_call.pop("attributions", self.attributions)
            inputs = merged_call.pop("inputs", self.inputs)
            show_sample_names = bool(
                merged_call.pop("show_sample_names", self.kwargs.get("show_sample_names", False))
            )
            sample_names_value = merged_call.pop("sample_names", self.kwargs.get("sample_names"))
            sample_names = _normalise_sample_names(sample_names_value)

            if sample_index is not None:
                attributions = attributions[sample_index : sample_index + 1]
                inputs = inputs[sample_index : sample_index + 1]
                if sample_names:
                    sample_names = sample_names[sample_index : sample_index + 1]
            else:
                limit = _batch_size(attributions) or _batch_size(inputs)
                if limit is not None:
                    sample_names = sample_names[:limit]

            context = VisualisationContext(
                algorithm=self.algorithm,
                sample_names=sample_names,
                show_sample_names=show_sample_names,
            )
            figure = vis.visualise(attributions, inputs=inputs, context=context, **merged_call)

            if (
                show_sample_names
                and sample_names
                and not figure.texts
                and not any(ax.get_title() for ax in figure.axes)
            ):
                figure.suptitle(_sample_names_title(sample_names), fontsize=10)
                figure.tight_layout()

            cls = type(vis)
            output_path = output_dir / f"{file_stem_prefix}_{cls.__name__}_{index}.png"
            try:
                figure.savefig(output_path, bbox_inches="tight", dpi=150)
            finally:
                plt.close(figure)
            selected_paths.append(output_path)

        return tuple(selected_paths)

    def log(
        self,
        tracker: BaseTracker | None,
        artifact_path: str = "transparency",
        use_subdirectory: bool = True,
        **kwargs: Any,
    ) -> None:
        if tracker is None:
            return

        target_path = self._log_target_path(
            artifact_path=artifact_path, use_subdirectory=use_subdirectory
        )

        if not self.visualiser_targets:
            tracker.log_artifacts(self.run_dir, target_subdirectory=target_path)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            staging_dir = Path(tmp_dir) / "explanation"
            staging_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.run_dir / "attributions.pt", staging_dir / "attributions.pt")
            (staging_dir / "metadata.json").write_text(
                json.dumps(self._metadata(visualiser_targets=[]), indent=2),
                encoding="utf-8",
            )
            tracker.log_artifacts(staging_dir, target_subdirectory=target_path)

    def _log_explainer_name(self) -> str:
        """
        Name used for the tracker artifact subdirectory.

        Keep fallback logic consistent with `VisualisationResult.log()` to avoid artifacts
        being split across different subdirectories when `explainer_name` is unset.
        """

        return self.explainer_name or self.run_dir.name

    def _log_target_path(self, *, artifact_path: str, use_subdirectory: bool) -> str:
        explainer_name = self._log_explainer_name()
        return f"{artifact_path}/{explainer_name}" if use_subdirectory else artifact_path


@dataclass
class VisualisationResult(Trackable):
    """PNG is written to ``output_path``; ``figure`` is closed after save to limit memory use."""

    explanation: ExplanationResult
    figure: Figure
    visualiser_name: str
    visualiser_target: str
    output_path: Path
    report_scope: str = "local"

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)

    def log(
        self,
        tracker: BaseTracker | None,
        artifact_path: str = "transparency",
        use_subdirectory: bool = True,
        **kwargs: Any,
    ) -> None:
        if tracker is None:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            explainer_name = self.explanation._log_explainer_name()
            staging_dir = Path(tmp_dir) / explainer_name
            staging_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.output_path, staging_dir / self.output_path.name)

            target_path = self.explanation._log_target_path(
                artifact_path=artifact_path,
                use_subdirectory=use_subdirectory,
            )
            tracker.log_artifacts(staging_dir, target_subdirectory=target_path)
