from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from hydra.core.hydra_config import HydraConfig

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..tracking.base_tracker import BaseTracker
    from .visualisers import BaseVisualiser


def _serialisable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {str(key): _serialisable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialisable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _serialisable(value.item())
        except Exception:
            pass
    return repr(value)


def resolve_default_run_dir() -> Path:
    try:
        return Path(HydraConfig.get().runtime.output_dir) / "transparency"
    except ValueError:
        return Path.cwd() / "transparency"


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

    def visualise(self, visualiser: BaseVisualiser, **kwargs: Any) -> VisualisationResult:
        figure = visualiser.visualise(self.attributions, inputs=self.inputs, **kwargs)
        visualiser_name = type(visualiser).__name__
        output_path = self.run_dir / f"{visualiser_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight", dpi=150)

        visualiser_target = f"{type(visualiser).__module__}.{visualiser_name}"
        if visualiser_target not in self.visualiser_targets:
            self.visualiser_targets.append(visualiser_target)
            self._write_metadata()

        return VisualisationResult(
            explanation=self,
            figure=figure,
            visualiser_name=visualiser_name,
            visualiser_target=visualiser_target,
            output_path=output_path,
        )

    def log(self, tracker: BaseTracker | None, artifact_path: str = "transparency") -> None:
        if tracker is None:
            return

        explainer_name = self.explainer_name or self.run_dir.name
        target_path = f"{artifact_path}/{explainer_name}"

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


@dataclass
class VisualisationResult:
    explanation: ExplanationResult
    figure: Figure
    visualiser_name: str
    visualiser_target: str
    output_path: Path

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)

    def log(self, tracker: BaseTracker | None, artifact_path: str = "transparency") -> None:
        if tracker is None:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            explainer_name = self.explanation.explainer_name or "default"
            staging_dir = Path(tmp_dir) / explainer_name
            staging_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.output_path, staging_dir / self.output_path.name)
            target_path = f"{artifact_path}/{explainer_name}"
            tracker.log_artifacts(staging_dir, target_subdirectory=target_path)
