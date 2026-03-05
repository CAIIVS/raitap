"""Integration tests for end-to-end workflows"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from omegaconf import OmegaConf

from raitap.configs.schema import AppConfig
from raitap.transparency import explain
from raitap.transparency.explainers import CaptumExplainer, ShapExplainer
from raitap.transparency.methods_registry import VisualiserIncompatibilityError
from raitap.transparency.visualisers import CaptumImageVisualiser, TabularBarChartVisualiser

# ---------------------------------------------------------------------------
# Algorithm / visualiser lists — extend here to widen coverage
# ---------------------------------------------------------------------------

# Captum algorithms that work without explicit baselines (zero baseline used)
CAPTUM_IMAGE_ALGORITHMS = [
    "IntegratedGradients",
    "Saliency",
    "DeepLift",
    "InputXGradient",
]

# SHAP algorithms compatible with ShapImageVisualiser (gradient-based only)
SHAP_IMAGE_ALGORITHMS = ["GradientExplainer", "DeepExplainer"]

# Captum algorithms suitable for tabular (2-D) inputs, no baselines needed
CAPTUM_TABULAR_ALGORITHMS = [
    "IntegratedGradients",
    "Saliency",
    "InputXGradient",
]

# SHAP algorithms compatible with tabular visualisers
SHAP_TABULAR_ALGORITHMS = ["GradientExplainer", "DeepExplainer"]

# SHAP tabular visualisers (all accept batch attributions)
_SHAP_TABULAR_VISUALISERS = [
    "raitap.transparency.ShapBarVisualiser",
    "raitap.transparency.ShapBeeswarmVisualiser",
    "raitap.transparency.ShapWaterfallVisualiser",
    "raitap.transparency.ShapForceVisualiser",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _captum_tc(**overrides):
    """Return an OmegaConf DictConfig mimicking the captum transparency config."""
    base = {
        "_target_": "raitap.transparency.CaptumExplainer",
        "algorithm": "IntegratedGradients",
        "visualisers": [{"_target_": "raitap.transparency.CaptumImageVisualiser"}],
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _shap_tc(**overrides):
    """Return an OmegaConf DictConfig mimicking the shap transparency config."""
    base = {
        "_target_": "raitap.transparency.ShapExplainer",
        "algorithm": "GradientExplainer",
        "visualisers": [{"_target_": "raitap.transparency.ShapImageVisualiser"}],
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _make_config(tmp_path, tc) -> AppConfig:
    """Wrap a transparency DictConfig in a minimal app config namespace."""
    return cast(
        AppConfig,
        SimpleNamespace(
            experiment_name="test",
            fallback_output_dir=str(tmp_path),
            transparency=tc,
        ),
    )


def _assert_explain_result(result, visualiser_name: str) -> None:
    """Common assertions for a successful explain() call."""
    assert "attributions" in result
    assert "visualisations" in result
    assert "run_dir" in result
    assert isinstance(result["attributions"], torch.Tensor)
    assert visualiser_name in result["visualisations"]
    assert (result["run_dir"] / "attributions.pt").exists()
    assert (result["run_dir"] / f"{visualiser_name}.png").exists()
    assert (result["run_dir"] / "metadata.json").exists()


# ---------------------------------------------------------------------------
# End-to-end workflows (direct use of explainer/visualiser classes)
# ---------------------------------------------------------------------------


class TestEndToEndWorkflows:
    """Tests using explainer classes directly (no explain() orchestrator)"""

    def test_end_to_end_captum(self, needs_captum, simple_cnn, sample_images, tmp_path):
        """Full workflow: create → compute attributions → visualize → save"""
        explainer = CaptumExplainer("IntegratedGradients")
        attributions = explainer.compute_attributions(simple_cnn, sample_images, target=0)
        assert isinstance(attributions, torch.Tensor)

        visualiser = CaptumImageVisualiser()
        output_path = tmp_path / "result.png"
        visualiser.save(attributions, output_path, inputs=sample_images)
        assert output_path.exists()

    def test_end_to_end_shap(self, needs_shap, simple_cnn, sample_images, tmp_path):
        """Full workflow with SHAP"""
        explainer = ShapExplainer("GradientExplainer")
        background = sample_images[:2]
        attributions = explainer.compute_attributions(
            simple_cnn, sample_images, background_data=background, target=0
        )
        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape == sample_images.shape

        visualiser = CaptumImageVisualiser()
        output_path = tmp_path / "shap_result.png"
        visualiser.save(attributions, output_path, inputs=sample_images)
        assert output_path.exists()

    def test_tabular_workflow(self, simple_mlp, sample_tabular, tmp_path):
        """Test tabular data workflow"""
        visualiser = TabularBarChartVisualiser(feature_names=[f"feature_{i}" for i in range(10)])
        attributions = torch.randn_like(sample_tabular)
        output_path = tmp_path / "tabular_result.png"
        visualiser.save(attributions, output_path)
        assert output_path.exists()

    def test_direct_instantiation(self, needs_captum, simple_cnn, sample_images):
        """Instantiate an explainer directly and compute attributions"""
        explainer = CaptumExplainer("Saliency")
        assert explainer is not None
        attributions = explainer.compute_attributions(simple_cnn, sample_images, target=0)
        assert isinstance(attributions, torch.Tensor)


# ---------------------------------------------------------------------------
# explain() orchestrator — image modality
# ---------------------------------------------------------------------------


class TestExplainCaptumImage:
    """explain() with parametrized captum algorithms + CaptumImageVisualiser."""

    @pytest.mark.parametrize("algorithm", CAPTUM_IMAGE_ALGORITHMS)
    def test_explain(self, needs_captum, algorithm, simple_cnn, sample_images, tmp_path):
        config = _make_config(tmp_path, _captum_tc(algorithm=algorithm))
        result = explain(config, simple_cnn, sample_images, target=0)
        _assert_explain_result(result, "CaptumImageVisualiser")
        assert result["attributions"].shape == sample_images.shape

    def test_explain_gradient_shap(self, needs_captum, simple_cnn, sample_images, tmp_path):
        """GradientShap requires explicit baselines — tested separately."""
        baselines = torch.zeros_like(sample_images)
        config = _make_config(tmp_path, _captum_tc(algorithm="GradientShap"))
        result = explain(config, simple_cnn, sample_images, target=0, baselines=baselines)
        _assert_explain_result(result, "CaptumImageVisualiser")


class TestExplainShapImage:
    """explain() with gradient-based SHAP algorithms + ShapImageVisualiser."""

    @pytest.mark.parametrize("algorithm", SHAP_IMAGE_ALGORITHMS)
    def test_explain(self, needs_shap, algorithm, simple_cnn, sample_images, tmp_path):
        config = _make_config(tmp_path, _shap_tc(algorithm=algorithm))
        result = explain(config, simple_cnn, sample_images, target=0)
        _assert_explain_result(result, "ShapImageVisualiser")


# ---------------------------------------------------------------------------
# explain() orchestrator — tabular modality
# ---------------------------------------------------------------------------


class TestExplainCaptumTabular:
    """explain() with parametrized captum algorithms + TabularBarChartVisualiser."""

    @pytest.mark.parametrize("algorithm", CAPTUM_TABULAR_ALGORITHMS)
    def test_explain(self, needs_captum, algorithm, simple_mlp, sample_tabular, tmp_path):
        config = _make_config(
            tmp_path,
            OmegaConf.create(
                {
                    "_target_": "raitap.transparency.CaptumExplainer",
                    "algorithm": algorithm,
                    "visualisers": [{"_target_": "raitap.transparency.TabularBarChartVisualiser"}],
                }
            ),
        )
        result = explain(config, simple_mlp, sample_tabular, target=0)
        _assert_explain_result(result, "TabularBarChartVisualiser")
        assert result["attributions"].shape == sample_tabular.shape


class TestExplainShapTabular:
    """explain() with SHAP algorithms x SHAP tabular visualisers (2x4 = 8 tests)."""

    @pytest.mark.parametrize("visualiser_target", _SHAP_TABULAR_VISUALISERS)
    @pytest.mark.parametrize("algorithm", SHAP_TABULAR_ALGORITHMS)
    def test_explain(
        self, needs_shap, algorithm, visualiser_target, simple_mlp, sample_tabular, tmp_path
    ):
        config = _make_config(
            tmp_path,
            OmegaConf.create(
                {
                    "_target_": "raitap.transparency.ShapExplainer",
                    "algorithm": algorithm,
                    "visualisers": [{"_target_": visualiser_target}],
                }
            ),
        )
        result = explain(config, simple_mlp, sample_tabular, target=0)
        visualiser_name = visualiser_target.rsplit(".", 1)[-1]
        _assert_explain_result(result, visualiser_name)


# ---------------------------------------------------------------------------
# Incompatible combinations
# ---------------------------------------------------------------------------


class TestIncompatibleCombinations:
    """explain() raises VisualiserIncompatibilityError for invalid algorithm/visualiser pairs."""

    @pytest.mark.parametrize("algorithm", ["KernelExplainer", "TreeExplainer"])
    def test_shap_image_raises(self, algorithm, simple_cnn, sample_images, tmp_path):
        """ShapImageVisualiser only accepts GradientExplainer and DeepExplainer."""
        config = _make_config(
            tmp_path,
            OmegaConf.create(
                {
                    "_target_": "raitap.transparency.ShapExplainer",
                    "algorithm": algorithm,
                    "visualisers": [{"_target_": "raitap.transparency.ShapImageVisualiser"}],
                }
            ),
        )
        with pytest.raises(VisualiserIncompatibilityError):
            explain(config, simple_cnn, sample_images)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestExplainErrors:
    def test_bad_target_raises(self, simple_cnn, sample_images, tmp_path):
        """explain() raises ValueError when _target_ can't be resolved."""
        config = _make_config(
            tmp_path,
            OmegaConf.create(
                {
                    "_target_": "raitap.transparency.NonExistentExplainer",
                    "algorithm": "IntegratedGradients",
                    "visualisers": [],
                }
            ),
        )
        with pytest.raises(ValueError, match="Could not instantiate explainer"):
            explain(config, simple_cnn, sample_images)
