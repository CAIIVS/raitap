from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import torch

from raitap.tracking.base_tracker import BaseTracker, Trackable
from raitap.utils.serialization import to_json_serialisable

from .contracts import (
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    ExplanationSemantics,
    OutputSpaceSpec,
    ScopeDefinitionStep,
    VisualisationContext,
    VisualSummarySpec,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .visualisers import BaseVisualiser


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


def _default_semantics() -> ExplanationSemantics:
    return ExplanationSemantics(
        scope=ExplanationScope.LOCAL,
        scope_definition_step=ScopeDefinitionStep.EXPLAINER_OUTPUT,
        payload_kind=ExplanationPayloadKind.ATTRIBUTIONS,
        method_families=frozenset(),
        target=None,
        sample_selection=None,
        input_spec=None,
        output_space=OutputSpaceSpec(
            space=ExplanationOutputSpace.INPUT_FEATURES,
            shape=None,
            layout=None,
        ),
    )


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


def _normalise_scope(scope: ExplanationScope | str) -> ExplanationScope:
    if isinstance(scope, ExplanationScope):
        return scope
    raw = str(scope).strip().lower()
    for candidate in ExplanationScope:
        if raw in {candidate.value, candidate.name.lower()}:
            return candidate
    raise ValueError(f"Unknown explanation scope {scope!r}.")


def _normalise_scope_definition_step(
    step: ScopeDefinitionStep | str,
) -> ScopeDefinitionStep:
    if isinstance(step, ScopeDefinitionStep):
        return step
    raw = str(step).strip().lower()
    for candidate in ScopeDefinitionStep:
        if raw in {candidate.value, candidate.name.lower()}:
            return candidate
    raise ValueError(f"Unknown scope definition step {step!r}.")


@dataclass(frozen=True)
class ConfiguredVisualiser:
    """Visualiser instance plus per-call kwargs for ``BaseVisualiser.visualise``."""

    visualiser: BaseVisualiser
    call_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationResult(Trackable):
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
    semantics: ExplanationSemantics = field(default_factory=_default_semantics)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if not isinstance(self.semantics, ExplanationSemantics):
            raise TypeError("ExplanationResult.semantics must be an ExplanationSemantics.")
        # Ensure tensors are detached and on CPU to avoid GPU memory retention
        self.attributions = self.attributions.detach().cpu()
        self.inputs = self.inputs.detach().cpu()

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
            "semantics": _serialisable_semantics(self.semantics),
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

            vis.validate_explanation(self, attributions, inputs)

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
                    scope=self._scope_for_visualiser(vis),
                    scope_definition_step=self._scope_definition_step_for_visualiser(vis),
                    visual_summary=getattr(type(vis), "visual_summary", None),
                )
            )

        if new_targets:
            self.visualiser_targets.extend(new_targets)
            self._write_metadata()

        return results

    def has_visualisations_for_scope(self, scope: ExplanationScope | str) -> bool:
        wanted = _normalise_scope(scope)
        return any(
            self._scope_for_visualiser(configured.visualiser) == wanted
            for configured in self.visualisers
        )

    def render_visualisations_for_scope(
        self,
        *,
        scope: ExplanationScope | str,
        sample_index: int | None = None,
    ) -> list[VisualisationResult]:
        """
        Render scoped visualisations without persisting them to disk.

        The returned ``VisualisationResult`` objects are intended for downstream
        consumers such as reporting, which own file staging and figure cleanup.
        Their ``output_path`` values are placeholders only and must not be treated
        as real on-disk artifacts or passed to ``VisualisationResult.log()``.
        """
        results: list[VisualisationResult] = []
        wanted = _normalise_scope(scope)

        for index, configured in enumerate(self.visualisers):
            vis = configured.visualiser
            if self._scope_for_visualiser(vis) != wanted:
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
            vis.validate_explanation(self, attributions, inputs)
            original_visualiser_sample_index = getattr(vis, "sample_index", None)
            reset_visualiser_sample_index = (
                sample_index is not None and original_visualiser_sample_index is not None
            )
            visualiser_with_sample_index: Any | None = None
            if reset_visualiser_sample_index:
                # Report rendering has already sliced to a one-sample batch, so visualisers
                # with their own batch selector must read index 0 inside that slice.
                visualiser_with_sample_index = vis
                visualiser_with_sample_index.sample_index = 0
            try:
                figure = vis.visualise(attributions, inputs=inputs, context=context, **merged_call)
            finally:
                if visualiser_with_sample_index is not None:
                    visualiser_with_sample_index.sample_index = original_visualiser_sample_index

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
            results.append(
                VisualisationResult(
                    explanation=self,
                    figure=figure,
                    visualiser_name=visualiser_name,
                    visualiser_target=f"{cls.__module__}.{visualiser_name}",
                    output_path=Path(visualiser_name).with_suffix(".png"),
                    scope=wanted,
                    scope_definition_step=self._scope_definition_step_for_visualiser(vis),
                    visual_summary=getattr(type(vis), "visual_summary", None),
                )
            )

        return results

    def _scope_for_visualiser(self, visualiser: BaseVisualiser) -> ExplanationScope:
        produced = getattr(type(visualiser), "produces_scope", None)
        if produced is not None:
            return _normalise_scope(produced)
        return self.semantics.scope

    def _scope_definition_step_for_visualiser(
        self,
        visualiser: BaseVisualiser,
    ) -> ScopeDefinitionStep:
        produced = getattr(type(visualiser), "produces_scope", None)
        if produced is None:
            return self.semantics.scope_definition_step
        step = getattr(type(visualiser), "scope_definition_step", None)
        if step is None:
            return ScopeDefinitionStep.VISUALISER_SUMMARY
        return _normalise_scope_definition_step(step)

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
    scope: ExplanationScope = ExplanationScope.LOCAL
    scope_definition_step: ScopeDefinitionStep = ScopeDefinitionStep.EXPLAINER_OUTPUT
    visual_summary: VisualSummarySpec | None = None

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        self.scope = _normalise_scope(self.scope)
        self.scope_definition_step = _normalise_scope_definition_step(self.scope_definition_step)

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
