"""
Centralised transparency E2E suite.

This module is the single home for heavy transparency combinations that run in
the PR-only E2E lane:

    # fast suite
    uv run pytest -m "not e2e"

    # heavy suite
    uv run pytest -m e2e -v --tb=long --mpl
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch
from omegaconf import OmegaConf

from raitap.models.backend import TorchBackend
from raitap.transparency import ExplanationResult, VisualisationResult
from raitap.transparency.explainers import ShapExplainer
from raitap.transparency.factory import Explanation
from raitap.transparency.results import ConfiguredVisualiser
from raitap.transparency.visualisers import (
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
)

if TYPE_CHECKING:
    from pathlib import Path

    import torch.nn as nn

    from raitap.configs.schema import AppConfig
    from raitap.models import Model

pytestmark = [pytest.mark.e2e]


def _read_metadata(run_dir: Path) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        json.loads((run_dir / "metadata.json").read_text(encoding="utf-8")),
    )


def _load_saved_attributions(run_dir: Path) -> torch.Tensor:
    return cast("torch.Tensor", torch.load(run_dir / "attributions.pt"))


def _assert_shap_metadata_invariants(
    metadata: dict[str, object],
    *,
    algorithm: str,
    experiment_name: str | None,
    has_visualisers: bool,
) -> None:
    assert set(metadata) == {"experiment_name", "target", "algorithm", "visualisers", "kwargs"}
    assert metadata["experiment_name"] == experiment_name
    assert metadata["algorithm"] == algorithm
    assert str(metadata["target"]).endswith("ShapExplainer")
    assert isinstance(metadata["kwargs"], dict)
    assert isinstance(metadata["visualisers"], list)
    if has_visualisers:
        assert len(cast("list[str]", metadata["visualisers"])) >= 1
    else:
        assert metadata["visualisers"] == []


def _captum_smoke_config(tmp_path: Path) -> AppConfig:
    return cast(
        "AppConfig",
        cast(
            "object",
            SimpleNamespace(
                experiment_name="test_captum_e2e",
                fallback_output_dir=str(tmp_path),
                transparency={
                    "captum_smoke": OmegaConf.create(
                        {
                            "_target_": "raitap.transparency.CaptumExplainer",
                            "algorithm": "IntegratedGradients",
                            "call": {"target": 0},
                            "visualisers": [
                                {
                                    "_target_": "raitap.transparency.CaptumImageVisualiser",
                                    "constructor": {
                                        "method": "heat_map",
                                        "show_colorbar": False,
                                        "include_original_image": False,
                                    },
                                }
                            ],
                        }
                    )
                },
            ),
        ),
    )


@pytest.mark.usefixtures("needs_captum")
def test_config_driven_captum_smoke_case(
    simple_cnn: torch.nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    config = _captum_smoke_config(tmp_path)
    model = cast("Model", SimpleNamespace(backend=TorchBackend(simple_cnn)))

    explanation = Explanation(config, "captum_smoke", model, sample_images)
    visualisations = explanation.visualise()
    metadata = _read_metadata(explanation.run_dir)
    saved_attributions = _load_saved_attributions(explanation.run_dir)

    assert isinstance(explanation, ExplanationResult)
    assert len(visualisations) == 1
    assert isinstance(visualisations[0], VisualisationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency" / "captum_smoke"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()
    assert torch.equal(saved_attributions, explanation.attributions)
    assert metadata["algorithm"] == "IntegratedGradients"
    assert str(metadata["target"]).endswith("CaptumExplainer")
    assert cast("dict[str, object]", metadata["kwargs"])["target"] == 0
    assert cast("list[str]", metadata["visualisers"])[0].endswith("CaptumImageVisualiser_0")
    assert visualisations[0].output_path == explanation.run_dir / "CaptumImageVisualiser_0.png"
    assert visualisations[0].output_path.exists()


@pytest.mark.usefixtures("needs_shap")
def test_deep_explainer_image_pipeline(
    simple_cnn: nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    explainer = ShapExplainer("DeepExplainer")
    background = sample_images[:2]

    explanation = explainer.explain(
        simple_cnn,
        sample_images,
        run_dir=tmp_path / "transparency",
        background_data=background,
        target=0,
        visualisers=[ConfiguredVisualiser(visualiser=ShapImageVisualiser())],
    )
    metadata_before_visualise = _read_metadata(explanation.run_dir)
    visualisations = explanation.visualise()
    metadata = _read_metadata(explanation.run_dir)
    saved_attributions = _load_saved_attributions(explanation.run_dir)

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()
    assert torch.equal(saved_attributions, explanation.attributions)
    _assert_shap_metadata_invariants(
        metadata_before_visualise,
        algorithm="DeepExplainer",
        experiment_name=None,
        has_visualisers=False,
    )
    _assert_shap_metadata_invariants(
        metadata,
        algorithm="DeepExplainer",
        experiment_name=None,
        has_visualisers=True,
    )
    assert cast("dict[str, object]", metadata["kwargs"])["target"] == 0
    assert cast("list[str]", metadata["visualisers"])[0].endswith("ShapImageVisualiser_0")
    assert len(visualisations) == 1
    assert visualisations[0].output_path == explanation.run_dir / "ShapImageVisualiser_0.png"
    assert visualisations[0].output_path.exists()


@pytest.mark.usefixtures("needs_shap")
def test_deep_explainer_multiclass_all_targets(
    simple_cnn: nn.Module,
    sample_images: torch.Tensor,
) -> None:
    explainer = ShapExplainer("DeepExplainer")
    background = sample_images[:2]

    all_class_attrs = explainer.compute_attributions(
        simple_cnn, sample_images, background_data=background
    )

    assert isinstance(all_class_attrs, torch.Tensor)
    assert all_class_attrs.shape[:-1] == sample_images.shape


@pytest.mark.usefixtures("needs_shap")
def test_gradient_explainer_per_sample_list_targets_beeswarm(
    simple_mlp: nn.Module,
    sample_tabular: torch.Tensor,
    tmp_path: Path,
) -> None:
    explainer = ShapExplainer("GradientExplainer")
    background = sample_tabular[:4]
    targets = [index % 2 for index in range(sample_tabular.shape[0])]

    explanation = explainer.explain(
        simple_mlp,
        sample_tabular,
        run_dir=tmp_path / "transparency",
        background_data=background,
        target=targets,
        visualisers=[
            ConfiguredVisualiser(
                visualiser=ShapBeeswarmVisualiser(
                    feature_names=[f"f{index}" for index in range(sample_tabular.shape[1])]
                )
            )
        ],
    )
    metadata_before_visualise = _read_metadata(explanation.run_dir)
    visualisations = explanation.visualise()
    metadata = _read_metadata(explanation.run_dir)
    saved_attributions = _load_saved_attributions(explanation.run_dir)

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == sample_tabular.shape
    assert torch.equal(saved_attributions, explanation.attributions)
    _assert_shap_metadata_invariants(
        metadata_before_visualise,
        algorithm="GradientExplainer",
        experiment_name=None,
        has_visualisers=False,
    )
    _assert_shap_metadata_invariants(
        metadata,
        algorithm="GradientExplainer",
        experiment_name=None,
        has_visualisers=True,
    )
    assert cast("dict[str, object]", metadata["kwargs"])["target"] == targets
    assert cast("list[str]", metadata["visualisers"])[0].endswith("ShapBeeswarmVisualiser_0")
    assert len(visualisations) == 1
    assert visualisations[0].output_path == explanation.run_dir / "ShapBeeswarmVisualiser_0.png"
    assert visualisations[0].output_path.exists()


@pytest.mark.usefixtures("needs_shap")
def test_gradient_explainer_tensor_target_indexing(
    simple_mlp: nn.Module,
    sample_tabular: torch.Tensor,
) -> None:
    explainer = ShapExplainer("GradientExplainer")
    background = sample_tabular[:4]
    target_tensor = torch.zeros(sample_tabular.shape[0], dtype=torch.long)

    attributions = explainer.compute_attributions(
        simple_mlp,
        sample_tabular,
        background_data=background,
        target=target_tensor,
    )

    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == sample_tabular.shape


@pytest.mark.usefixtures("needs_shap")
def test_gradient_explainer_waterfall_visualiser(
    simple_mlp: nn.Module,
    sample_tabular: torch.Tensor,
    tmp_path: Path,
) -> None:
    explainer = ShapExplainer("GradientExplainer")
    background = sample_tabular[:4]

    explanation = explainer.explain(
        simple_mlp,
        sample_tabular,
        run_dir=tmp_path / "transparency",
        background_data=background,
        target=0,
        visualisers=[
            ConfiguredVisualiser(
                visualiser=ShapWaterfallVisualiser(
                    feature_names=[f"f{index}" for index in range(sample_tabular.shape[1])],
                    expected_value=0.5,
                    sample_index=1,
                    max_display=5,
                )
            )
        ],
    )
    metadata_before_visualise = _read_metadata(explanation.run_dir)
    visualisations = explanation.visualise()
    metadata = _read_metadata(explanation.run_dir)

    assert len(visualisations) == 1
    _assert_shap_metadata_invariants(
        metadata_before_visualise,
        algorithm="GradientExplainer",
        experiment_name=None,
        has_visualisers=False,
    )
    _assert_shap_metadata_invariants(
        metadata,
        algorithm="GradientExplainer",
        experiment_name=None,
        has_visualisers=True,
    )
    assert cast("dict[str, object]", metadata["kwargs"])["target"] == 0
    assert cast("list[str]", metadata["visualisers"])[0].endswith("ShapWaterfallVisualiser_0")
    assert visualisations[0].output_path == explanation.run_dir / "ShapWaterfallVisualiser_0.png"
    assert visualisations[0].output_path.exists()


@pytest.mark.usefixtures("needs_shap")
def test_gradient_explainer_bar_visualiser_with_inputs(
    simple_mlp: nn.Module,
    sample_tabular: torch.Tensor,
    tmp_path: Path,
) -> None:
    explainer = ShapExplainer("GradientExplainer")
    background = sample_tabular[:4]

    explanation = explainer.explain(
        simple_mlp,
        sample_tabular,
        run_dir=tmp_path / "transparency",
        background_data=background,
        target=0,
        visualisers=[
            ConfiguredVisualiser(
                visualiser=ShapBarVisualiser(
                    feature_names=[f"f{index}" for index in range(sample_tabular.shape[1])]
                )
            )
        ],
    )
    metadata_before_visualise = _read_metadata(explanation.run_dir)
    visualisations = explanation.visualise()
    metadata = _read_metadata(explanation.run_dir)

    assert len(visualisations) == 1
    _assert_shap_metadata_invariants(
        metadata_before_visualise,
        algorithm="GradientExplainer",
        experiment_name=None,
        has_visualisers=False,
    )
    _assert_shap_metadata_invariants(
        metadata,
        algorithm="GradientExplainer",
        experiment_name=None,
        has_visualisers=True,
    )
    assert cast("dict[str, object]", metadata["kwargs"])["target"] == 0
    assert cast("list[str]", metadata["visualisers"])[0].endswith("ShapBarVisualiser_0")
    assert visualisations[0].output_path == explanation.run_dir / "ShapBarVisualiser_0.png"
    assert visualisations[0].output_path.exists()


@pytest.mark.usefixtures("needs_shap")
def test_explanation_factory_shap_gradient_full_pipeline(
    simple_cnn: nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    config = cast(
        "AppConfig",
        cast(
            "object",
            SimpleNamespace(
                experiment_name="test_shap_e2e",
                fallback_output_dir=str(tmp_path),
                transparency={
                    "shap_gradient": OmegaConf.create(
                        {
                            "_target_": "raitap.transparency.ShapExplainer",
                            "algorithm": "GradientExplainer",
                            "call": {"target": 0},
                            "visualisers": [
                                {"_target_": "raitap.transparency.ShapImageVisualiser"}
                            ],
                        }
                    )
                },
            ),
        ),
    )

    model = cast("Model", SimpleNamespace(backend=TorchBackend(simple_cnn)))
    explanation = Explanation(
        config,
        "shap_gradient",
        model,
        sample_images,
        background_data=sample_images[:2],
    )

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency" / "shap_gradient"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()
    saved_attributions = _load_saved_attributions(explanation.run_dir)
    metadata_before_visualise = _read_metadata(explanation.run_dir)
    assert torch.equal(saved_attributions, explanation.attributions)
    _assert_shap_metadata_invariants(
        metadata_before_visualise,
        algorithm="GradientExplainer",
        experiment_name="test_shap_e2e",
        has_visualisers=False,
    )
    assert cast("dict[str, object]", metadata_before_visualise["kwargs"])["target"] == 0

    visualisations = explanation.visualise()
    assert len(visualisations) == 1
    metadata = _read_metadata(explanation.run_dir)
    _assert_shap_metadata_invariants(
        metadata,
        algorithm="GradientExplainer",
        experiment_name="test_shap_e2e",
        has_visualisers=True,
    )
    assert cast("list[str]", metadata["visualisers"])[0].endswith("ShapImageVisualiser_0")
    assert visualisations[0].output_path == explanation.run_dir / "ShapImageVisualiser_0.png"
    assert visualisations[0].output_path.exists()


@pytest.mark.usefixtures("needs_shap")
def test_end_to_end_shap_object_api(
    simple_cnn: nn.Module,
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
        visualisers=[ConfiguredVisualiser(visualiser=ShapImageVisualiser())],
    )
    visualisations = explanation.visualise()
    metadata = _read_metadata(explanation.run_dir)
    saved_attributions = _load_saved_attributions(explanation.run_dir)

    assert isinstance(explanation, ExplanationResult)
    assert len(visualisations) == 1
    assert isinstance(visualisations[0], VisualisationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()
    assert saved_attributions.shape == sample_images.shape
    assert metadata["algorithm"] == "GradientExplainer"
    assert str(metadata["target"]).endswith("ShapExplainer")
    assert cast("dict[str, object]", metadata["kwargs"])["target"] == 0
    assert cast("list[str]", metadata["visualisers"]) == [
        "raitap.transparency.visualisers.shap_visualisers.ShapImageVisualiser_0"
    ]
    assert visualisations[0].output_path == explanation.run_dir / "ShapImageVisualiser_0.png"
    assert visualisations[0].output_path.exists()
