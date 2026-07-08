"""Integration tests for the object-based transparency API."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import pytest
from omegaconf import OmegaConf

from raitap.configs import set_output_root
from raitap.configs.schema import DataConfig
from raitap.models.access import explanation_model
from raitap.models.torch_backend import TorchBackend
from raitap.transparency import ExplanationResult, VisualisationResult
from raitap.transparency.contracts import InputSpec
from raitap.transparency.explainers import CaptumExplainer
from raitap.transparency.phase import prepare_explainer
from raitap.transparency.results import ConfiguredVisualiser
from raitap.transparency.visualisers import (
    CaptumImageVisualiser,
    TabularBarChartVisualiser,
)

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
        cast(
            "object",
            SimpleNamespace(
                experiment_name="test",
                _output_root="unused",
                transparency={
                    "test_explainer": OmegaConf.create(
                        {
                            "use": "captum",
                            "algorithm": "IntegratedGradients",
                            "call": {"target": 0},
                            "visualisers": [{"use": "captum_image"}],
                        }
                    )
                },
                # ``resolve_per_image_transform``'s no-``resolved_preprocessing``
                # fallback reads ``config.data`` directly; a real (defaulted)
                # ``DataConfig`` keeps that read honest instead of re-adding a
                # ``getattr`` default here.
                data=DataConfig(),
            ),
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
        raitap_kwargs={
            "input_metadata": InputSpec(
                kind="image",
                shape=tuple(sample_images.shape),
                layout="NCHW",
                metadata={"kind": "image", "layout": "NCHW"},
            )
        },
        visualisers=[ConfiguredVisualiser(visualiser=CaptumImageVisualiser())],
    )
    visualisations = explanation._visualise()

    assert isinstance(explanation, ExplanationResult)
    assert len(visualisations) == 1
    assert isinstance(visualisations[0], VisualisationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()
    assert visualisations[0].output_path == explanation.run_dir / "CaptumImageVisualiser_0.png"
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
        raitap_kwargs={
            "input_metadata": InputSpec(
                kind="tabular",
                shape=tuple(sample_tabular.shape),
                layout="(B,F)",
                metadata={"kind": "tabular", "layout": "(B,F)"},
            )
        },
    )

    explanation.visualisers = [
        ConfiguredVisualiser(
            visualiser=TabularBarChartVisualiser(feature_names=[f"feature_{i}" for i in range(10)])
        )
    ]
    visualisations = explanation._visualise()

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
    set_output_root(config, tmp_path)

    model = SimpleNamespace(backend=TorchBackend(simple_cnn))
    config.model = SimpleNamespace(class_names=None)  # type: ignore[attr-defined]
    data = SimpleNamespace(tensor=sample_images, sample_ids=None)
    prepared = prepare_explainer(
        config,
        "test_explainer",
        cast("Any", model),
        resolved_preprocessing=None,
        input_metadata=InputSpec(
            kind="image",
            shape=tuple(sample_images.shape),
            layout="NCHW",
            metadata={"kind": "image", "layout": "NCHW"},
        ),
        data=cast("Any", data),
    )
    backend = cast("Any", prepared.backend)
    explanation: ExplanationResult = cast(
        "ExplanationResult",
        prepared.explainer.explain(  # type: ignore[attr-defined]
            explanation_model(backend, prepared.explainer),
            sample_images,
            backend=backend,
            run_dir=prepared.base_run_dir,
            experiment_name=prepared.experiment_name,
            explainer_target=prepared.explainer_target,
            explainer_name=prepared.name,
            visualisers=prepared.visualisers,
            raitap_kwargs=prepared.raitap_kwargs,
            call_provenance=prepared.call_provenance,
            **prepared.merged_kwargs,
        ),
    )
    visualisations = explanation._visualise()

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
        raitap_kwargs={
            "input_metadata": InputSpec(
                kind="image",
                shape=tuple(sample_images.shape),
                layout="NCHW",
                metadata={"kind": "image", "layout": "NCHW"},
            )
        },
        visualisers=[ConfiguredVisualiser(visualiser=CaptumImageVisualiser())],
    )
    _ = explanation._visualise()

    explanation.log(tracker, use_subdirectory=False)  # type: ignore[arg-type]

    assert len(tracker.logged_directories) == 1
    logged_directory = tracker.logged_directories[0]
    assert logged_directory["artifact_path"] == "transparency"
    assert logged_directory["files"] == ["attributions.pt", "metadata.json"]
    assert "metadata" in logged_directory
    assert logged_directory["metadata"]["visualisers"] == []


def test_tree_explainer_end_to_end_xgboost(tmp_path: Path) -> None:
    import numpy as np
    import torch

    pytest.importorskip("shap")
    xgboost = pytest.importorskip("xgboost")

    from raitap.models.xgboost_backend import XGBoostBackend
    from raitap.transparency.explainers import ShapExplainer
    from raitap.transparency.visualisers import ShapBarVisualiser

    rng = np.random.default_rng(0)
    feature_count = 6
    features = rng.normal(size=(32, feature_count)).astype(np.float32)
    labels = (features.sum(axis=1) > 0).astype(int)
    clf = xgboost.XGBClassifier(n_estimators=8, max_depth=3)
    clf.fit(features, labels)
    model_path = tmp_path / "model.ubj"
    clf.save_model(model_path)

    backend = XGBoostBackend.from_path(model_path, model_cfg=None, hardware="cpu")
    explainer = ShapExplainer("TreeExplainer")
    model = explanation_model(backend, explainer)  # -> the fitted estimator

    sample = torch.from_numpy(features[:8])
    explanation = explainer.explain(
        model,
        sample,
        backend=backend,
        run_dir=tmp_path / "transparency",
        target=1,
        raitap_kwargs={
            "input_metadata": InputSpec(
                kind="tabular",
                shape=tuple(sample.shape),
                layout="(B,F)",
                metadata={"kind": "tabular", "layout": "(B,F)"},
            )
        },
        visualisers=[
            ConfiguredVisualiser(
                visualiser=ShapBarVisualiser(feature_names=[f"f{i}" for i in range(feature_count)])
            )
        ],
    )
    visualisations = explanation._visualise()

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == (sample.shape[0], feature_count)
    assert (explanation.run_dir / "attributions.pt").exists()
    assert len(visualisations) == 1
    assert visualisations[0].output_path.exists()


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
        raitap_kwargs={
            "input_metadata": InputSpec(
                kind="image",
                shape=tuple(sample_images.shape),
                layout="NCHW",
                metadata={"kind": "image", "layout": "NCHW"},
            )
        },
        visualisers=[ConfiguredVisualiser(visualiser=CaptumImageVisualiser())],
    )
    visualisations = explanation._visualise()

    assert len(visualisations) == 1
    visualisations[0].log(tracker, use_subdirectory=False)  # type: ignore[arg-type]

    assert len(tracker.logged_directories) == 1
    logged_directory = tracker.logged_directories[0]
    assert logged_directory["artifact_path"] == "transparency"
    assert logged_directory["files"] == ["CaptumImageVisualiser_0.png"]
    assert "metadata" not in logged_directory
