"""Integration tests for the object-based transparency API."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NotRequired, TypedDict, cast

from omegaconf import OmegaConf

from raitap.configs.schema import AppConfig
from raitap.tracking.base import Tracker
from raitap.transparency import ExplanationResult, VisualisationResult
from raitap.transparency.explainers import CaptumExplainer, ShapExplainer
from raitap.transparency.factory import Explanation, create_visualisers
from raitap.transparency.visualisers import CaptumImageVisualiser, TabularBarChartVisualiser


class LoggedDirectory(TypedDict):
    artifact_path: str
    files: list[str]
    metadata: NotRequired[dict[str, Any]]


class RecordingTracker:
    def __init__(self) -> None:
        self.logged_directories: list[LoggedDirectory] = []

    def start_assessment(self, assessment_name: str) -> None:
        return None

    def log_config(self, config: Any) -> None:
        return None

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        return None

    def log_dataset(self, dataset_info: dict[str, Any], artifact_path: str = "dataset") -> None:
        return None

    def log_artifacts(self, local_dir: str | Path, artifact_path: str) -> None:
        directory = Path(local_dir)
        files = sorted(path.name for path in directory.iterdir())
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

    def finalize(self, status: str = "FINISHED") -> None:
        return None


def _captum_config() -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            experiment_name="test",
            fallback_output_dir="unused",
            explainers={
                "test_explainer": OmegaConf.create(
                    {
                        "_target_": "raitap.transparency.CaptumExplainer",
                        "algorithm": "IntegratedGradients",
                        "visualisers": [{"_target_": "raitap.transparency.CaptumImageVisualiser"}],
                    }
                )
            },
        ),
    )


def test_end_to_end_captum_object_api(needs_captum, simple_cnn, sample_images, tmp_path):
    explainer = CaptumExplainer("IntegratedGradients")

    explanation = explainer.explain(
        simple_cnn,
        sample_images,
        run_dir=tmp_path / "transparency",
        target=0,
    )
    visualisation = explanation.visualise(CaptumImageVisualiser())

    assert isinstance(explanation, ExplanationResult)
    assert isinstance(visualisation, VisualisationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()
    assert visualisation.output_path == explanation.run_dir / "CaptumImageVisualiser.png"
    assert visualisation.output_path.exists()


def test_end_to_end_shap_object_api(needs_shap, simple_cnn, sample_images, tmp_path):
    explainer = ShapExplainer("GradientExplainer")

    explanation = explainer.explain(
        simple_cnn,
        sample_images,
        run_dir=tmp_path / "transparency",
        background_data=sample_images[:2],
        target=0,
    )
    visualisation = explanation.visualise(CaptumImageVisualiser())

    assert isinstance(explanation, ExplanationResult)
    assert isinstance(visualisation, VisualisationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert visualisation.output_path.exists()


def test_tabular_visualisation_object_api(simple_mlp, sample_tabular, tmp_path):
    explainer = CaptumExplainer("Saliency")
    explanation = explainer.explain(
        simple_mlp,
        sample_tabular,
        run_dir=tmp_path / "transparency",
        target=0,
    )

    visualisation = explanation.visualise(
        TabularBarChartVisualiser(feature_names=[f"feature_{i}" for i in range(10)])
    )

    assert isinstance(visualisation, VisualisationResult)
    assert visualisation.output_path.exists()


def test_config_helpers_support_visualiser_for_loop(
    needs_captum, simple_cnn, sample_images, tmp_path
):
    config = _captum_config()
    config.fallback_output_dir = str(tmp_path)

    explanation = Explanation(config, "test_explainer", simple_cnn, sample_images, target=0)
    visualisations = [
        explanation.visualise(visualiser)
        for visualiser in create_visualisers(config.explainers["test_explainer"])
    ]

    assert isinstance(explanation, ExplanationResult)
    assert len(visualisations) == 1
    assert isinstance(visualisations[0], VisualisationResult)
    assert visualisations[0].output_path.exists()


def test_explanation_log_only_uploads_explanation_artifacts(
    needs_captum, simple_cnn, sample_images, tmp_path
):
    tracker = RecordingTracker()
    explanation = CaptumExplainer("IntegratedGradients").explain(
        simple_cnn,
        sample_images,
        run_dir=tmp_path / "transparency",
        target=0,
    )
    explanation.visualise(CaptumImageVisualiser())

    explanation.log(cast("Tracker", tracker))

    assert len(tracker.logged_directories) == 1
    logged_directory = tracker.logged_directories[0]
    assert logged_directory["artifact_path"] == "transparency"
    assert logged_directory["files"] == ["attributions.pt", "metadata.json"]
    assert "metadata" in logged_directory
    assert logged_directory["metadata"]["visualisers"] == []


def test_visualisation_log_uploads_only_visualisation_artifact(
    needs_captum, simple_cnn, sample_images, tmp_path
):
    tracker = RecordingTracker()
    explanation = CaptumExplainer("IntegratedGradients").explain(
        simple_cnn,
        sample_images,
        run_dir=tmp_path / "transparency",
        target=0,
    )
    visualisation = explanation.visualise(CaptumImageVisualiser())

    visualisation.log(tracker)

    assert len(tracker.logged_directories) == 1
    logged_directory = tracker.logged_directories[0]
    assert logged_directory["artifact_path"] == "transparency"
    assert logged_directory["files"] == ["CaptumImageVisualiser.png"]
    assert "metadata" not in logged_directory
