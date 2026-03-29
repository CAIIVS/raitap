from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import torch
from hydra.core.hydra_config import HydraConfig

from raitap.utils.serialization import to_json_serialisable

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..tracking.base_tracker import BaseTracker
    from .visualisers import BaseVisualiser


def _serialisable(value: Any) -> Any:
    return to_json_serialisable(value)


def resolve_default_run_dir() -> Path:
    try:
        return Path(HydraConfig.get().runtime.output_dir) / "transparency"
    except ValueError:
        return Path.cwd() / "transparency"


@dataclass(frozen=True)
class ConfiguredVisualiser:
    """Visualiser instance plus per-call kwargs for ``BaseVisualiser.visualise``."""

    visualiser: BaseVisualiser
    call_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationResult:
    attributions: torch.Tensor
    inputs: torch.Tensor
    run_dir: Path
    experiment_name: str | None
    explainer_target: str
    algorithm: str
    explainer_name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    visualiser_targets: list[str] = field(default_factory=list)
    visualisers: list[ConfiguredVisualiser] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)

    def write_artifacts(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.attributions, self.run_dir / "attributions.pt")
        self._write_metadata()

    def _metadata(self, *, visualiser_targets: list[str] | None = None) -> dict[str, Any]:
        targets = self.visualiser_targets if visualiser_targets is None else visualiser_targets
        return {
            "experiment_name": self.experiment_name,
            "target": self.explainer_target,
            "algorithm": self.algorithm,
            "visualisers": targets,
            "kwargs": {key: _serialisable(value) for key, value in self.kwargs.items()},
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
            figure = vis.visualise(attributions, inputs=inputs, **merged_call)
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
                )
            )

        if new_targets:
            self.visualiser_targets.extend(new_targets)
            self._write_metadata()

        return results

    def log(
        self,
        tracker: BaseTracker | None,
        artifact_path: str = "transparency",
        use_subdirectory: bool = True,
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
class VisualisationResult:
    """PNG is written to ``output_path``; ``figure`` is closed after save to limit memory use."""

    explanation: ExplanationResult
    figure: Figure
    visualiser_name: str
    visualiser_target: str
    output_path: Path

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)

    def log(
        self,
        tracker: BaseTracker | None,
        artifact_path: str = "transparency",
        use_subdirectory: bool = True,
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
