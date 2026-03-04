"""Integration tests for end-to-end workflows"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from raitap.transparency import explain
from raitap.transparency.explainers import CaptumExplainer, ShapExplainer
from raitap.transparency.methods_registry import VisualiserIncompatibilityError
from raitap.transparency.visualisers import CaptumImageVisualiser, TabularBarChartVisualiser

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


def _make_config(tmp_path, tc):
    """Wrap a transparency DictConfig in a minimal app config namespace."""
    return SimpleNamespace(
        experiment_name="test",
        fallback_output_dir=str(tmp_path),
        transparency=tc,
    )


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
# explain() orchestrator
# ---------------------------------------------------------------------------


class TestExplainFunction:
    """Test the top-level explain() API"""

    def test_explain_captum_image(self, needs_captum, simple_cnn, sample_images, tmp_path):
        """explain() with captum + image visualiser"""
        config = _make_config(tmp_path, _captum_tc())
        result = explain(config, simple_cnn, sample_images, target=0)

        assert "attributions" in result
        assert "visualisations" in result
        assert "run_dir" in result
        assert isinstance(result["attributions"], torch.Tensor)
        assert "CaptumImageVisualiser" in result["visualisations"]
        run_dir = result["run_dir"]
        assert (run_dir / "attributions.pt").exists()
        assert (run_dir / "CaptumImageVisualiser.png").exists()
        assert (run_dir / "metadata.json").exists()

    def test_explain_incompatible_visualiser_raises(self, simple_cnn, sample_images, tmp_path):
        """explain() raises VisualiserIncompatibilityError for SHAP image + KernelExplainer"""
        config = _make_config(
            tmp_path,
            OmegaConf.create(
                {
                    "_target_": "raitap.transparency.ShapExplainer",
                    "algorithm": "KernelExplainer",
                    "visualisers": [{"_target_": "raitap.transparency.ShapImageVisualiser"}],
                }
            ),
        )
        with pytest.raises(VisualiserIncompatibilityError):
            explain(config, simple_cnn, sample_images)

    def test_explain_bad_target_raises(self, simple_cnn, sample_images, tmp_path):
        """explain() raises ValueError when _target_ can't be resolved"""
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
