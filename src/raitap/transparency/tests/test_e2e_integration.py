"""Integration tests for the object-based transparency API."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import pytest
from omegaconf import OmegaConf

from raitap.transparency import ExplanationResult, VisualisationResult
from raitap.transparency.explainers import CaptumExplainer, ShapExplainer
from raitap.transparency.factory import Explanation
from raitap.transparency.results import ConfiguredVisualiser
from raitap.transparency.visualisers import CaptumImageVisualiser, TabularBarChartVisualiser

if TYPE_CHECKING:
    import torch

    from raitap.configs.schema import AppConfig


class LoggedDirectory(TypedDict):
    artifact_path: str
    files: list[str]
    metadata: NotRequired[dict[str, Any]]


class RecordingTracker:
    def __init__(self) -> None:
        self.logged_directories: list[LoggedDirectory] = []

    def log_config(self) -> None:
        return None

    def log_model(self, model: Any) -> None:
        return None

    def log_dataset(self, dataset_info: dict[str, Any]) -> None:
        return None

    def log_artifacts(
        self, source_directory: str | Path | None, target_subdirectory: str | None = None
    ) -> None:
        directory = Path(source_directory) if source_directory else Path()
        files = sorted(path.name for path in directory.iterdir())
        artifact_path = target_subdirectory or ""
        logged_directory: LoggedDirectory = {
            "artifact_path": artifact_path,
            "files": files,
        }
        if "metadata.json" in files:
            logged_directory["metadata"] = cast(
                "dict[str, Any]",
                json.loads((directory / "metadata.json").read_text(encoding="utf-8")),
            )
        self.logged_directories.append(logged_directory)

    def log_metrics(
        self,
        metrics: dict[str, float | int | bool],
        prefix: str = "performance",
    ) -> None:
        return None

    def terminate(self, successfully: bool = True) -> None:
        return None


def _captum_config() -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            experiment_name="test",
            fallback_output_dir="unused",
            transparency={
                "test_explainer": OmegaConf.create(
                    {
                        "_target_": "raitap.transparency.CaptumExplainer",
                        "algorithm": "IntegratedGradients",
                        "call": {"target": 0},
                        "visualisers": [{"_target_": "raitap.transparency.CaptumImageVisualiser"}],
                    }
                )
            },
        ),
    )


@pytest.mark.usefixtures("needs_captum")
def test_end_to_end_captum_object_api(
    simple_cnn: torch.nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    explainer = CaptumExplainer("IntegratedGradients")

    explanation = explainer.explain(
        simple_cnn,
        sample_images,
        run_dir=tmp_path / "transparency",
        target=0,
        visualisers=[ConfiguredVisualiser(visualiser=CaptumImageVisualiser())],
    )
    visualisations = explanation.visualise()

    assert isinstance(explanation, ExplanationResult)
    assert len(visualisations) == 1
    assert isinstance(visualisations[0], VisualisationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()
    assert visualisations[0].output_path == explanation.run_dir / "CaptumImageVisualiser_0.png"
    assert visualisations[0].output_path.exists()


@pytest.mark.usefixtures("needs_shap", "needs_captum")
def test_end_to_end_shap_object_api(
    simple_cnn: torch.nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    explainer = ShapExplainer("GradientExplainer")

    explanation = explainer.explain(
        simple_cnn,
        sample_images,
        run_dir=tmp_path / "transparency",
        background_data=sample_images[:2],
        target=0,
        visualisers=[ConfiguredVisualiser(visualiser=CaptumImageVisualiser())],
    )
    visualisations = explanation.visualise()

    assert isinstance(explanation, ExplanationResult)
    assert len(visualisations) == 1
    assert isinstance(visualisations[0], VisualisationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert visualisations[0].output_path.exists()


@pytest.mark.usefixtures("needs_captum")
def test_tabular_visualisation_object_api(
    simple_mlp: torch.nn.Module,
    sample_tabular: torch.Tensor,
    tmp_path: Path,
) -> None:
    explainer = CaptumExplainer("Saliency")
    explanation = explainer.explain(
        simple_mlp,
        sample_tabular,
        run_dir=tmp_path / "transparency",
        target=0,
    )

    explanation.visualisers = [
        ConfiguredVisualiser(
            visualiser=TabularBarChartVisualiser(feature_names=[f"feature_{i}" for i in range(10)])
        )
    ]
    visualisations = explanation.visualise()

    assert len(visualisations) == 1
    assert isinstance(visualisations[0], VisualisationResult)
    assert visualisations[0].output_path.exists()


@pytest.mark.usefixtures("needs_captum")
def test_config_helpers_support_visualiser_for_loop(
    simple_cnn: torch.nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    config = _captum_config()
    config.fallback_output_dir = str(tmp_path)

    model = SimpleNamespace(network=simple_cnn)
    explanation: ExplanationResult = Explanation(
        config,
        "test_explainer",
        model,  # type: ignore[arg-type]
        sample_images,
        target=0,
    )
    visualisations = explanation.visualise()

    assert isinstance(explanation, ExplanationResult)
    assert len(visualisations) == 1
    assert isinstance(visualisations[0], VisualisationResult)
    assert visualisations[0].output_path.exists()


@pytest.mark.usefixtures("needs_captum")
def test_explanation_log_only_uploads_explanation_artifacts(
    simple_cnn: torch.nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    tracker = RecordingTracker()
    explanation = CaptumExplainer("IntegratedGradients").explain(
        simple_cnn,
        sample_images,
        run_dir=tmp_path / "transparency",
        target=0,
        visualisers=[ConfiguredVisualiser(visualiser=CaptumImageVisualiser())],
    )
    _ = explanation.visualise()

    explanation.log(tracker, use_subdirectory=False)  # type: ignore[arg-type]

    assert len(tracker.logged_directories) == 1
    logged_directory = tracker.logged_directories[0]
    assert logged_directory["artifact_path"] == "transparency"
    assert logged_directory["files"] == ["attributions.pt", "metadata.json"]
    assert "metadata" in logged_directory
    assert logged_directory["metadata"]["visualisers"] == []


@pytest.mark.usefixtures("needs_captum")
def test_visualisation_log_uploads_only_visualisation_artifact(
    simple_cnn: torch.nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    tracker = RecordingTracker()
    explanation = CaptumExplainer("IntegratedGradients").explain(
        simple_cnn,
        sample_images,
        run_dir=tmp_path / "transparency",
        target=0,
        visualisers=[ConfiguredVisualiser(visualiser=CaptumImageVisualiser())],
    )
    visualisations = explanation.visualise()

    assert len(visualisations) == 1
    visualisations[0].log(tracker, use_subdirectory=False)  # type: ignore[arg-type]

    assert len(tracker.logged_directories) == 1
    logged_directory = tracker.logged_directories[0]
    assert logged_directory["artifact_path"] == "transparency"
    assert logged_directory["files"] == ["CaptumImageVisualiser_0.png"]
    assert "metadata" not in logged_directory
