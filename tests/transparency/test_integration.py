"""Integration tests for end-to-end workflows"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from raitap.transparency import create_explainer, explain, method_from_config
from raitap.transparency.methods import SHAP, Captum
from raitap.transparency.visualisers import ImageHeatmapvisualiser, TabularBarChartvisualiser


class TestFactoryFunction:
    """Test create_explainer factory"""

    def test_create_captum_explainer(self):
        """Test creating Captum explainer"""
        explainer = create_explainer(Captum.IntegratedGradients)
        assert explainer is not None

    def test_create_shap_explainer(self):
        """Test creating SHAP explainer"""
        explainer = create_explainer(SHAP.GradientExplainer)
        assert explainer is not None

    def test_method_from_config_captum(self):
        """Test config bridge for Captum"""
        config = SimpleNamespace(framework="captum", algorithm="IntegratedGradients")
        method = method_from_config(config)
        assert method.framework == "captum"
        assert method.algorithm == "IntegratedGradients"

    def test_method_from_config_shap(self):
        """Test config bridge for SHAP"""
        config = SimpleNamespace(framework="shap", algorithm="GradientExplainer")
        method = method_from_config(config)
        assert method.framework == "shap"
        assert method.algorithm == "GradientExplainer"

    def test_method_from_config_invalid_framework(self):
        """Test error for invalid framework"""
        config = SimpleNamespace(framework="invalid", algorithm="SomeMethod")
        with pytest.raises(ValueError) as exc_info:
            method_from_config(config)
        assert "Unknown framework" in str(exc_info.value)

    def test_method_from_config_invalid_algorithm(self):
        """Test error for invalid algorithm"""
        config = SimpleNamespace(framework="captum", algorithm="InvalidMethod")
        with pytest.raises(ValueError) as exc_info:
            method_from_config(config)
        assert "InvalidMethod" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)


class TestEndToEndWorkflows:
    """Test complete workflows"""

    @pytest.mark.skipif(
        not pytest.importorskip("captum", reason="Captum not installed"),
        reason="Captum not available",
    )
    def test_end_to_end_captum(self, simple_cnn, sample_images, tmp_path):
        """Full workflow: create → compute attributions → visualize → save"""
        explainer = create_explainer(Captum.IntegratedGradients)
        attributions = explainer.compute_attributions(simple_cnn, sample_images, target=0)
        assert isinstance(attributions, torch.Tensor)

        visualiser = ImageHeatmapvisualiser()
        output_path = tmp_path / "result.png"
        visualiser.save(attributions, output_path, inputs=sample_images)
        assert output_path.exists()

    @pytest.mark.skipif(
        not pytest.importorskip("shap", reason="SHAP not installed"), reason="SHAP not available"
    )
    def test_end_to_end_shap(self, simple_cnn, sample_images, tmp_path):
        """Full workflow with SHAP"""
        explainer = create_explainer(SHAP.GradientExplainer)
        background = sample_images[:2]
        attributions = explainer.compute_attributions(
            simple_cnn, sample_images, background_data=background, target=0
        )
        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape == sample_images.shape

        visualiser = ImageHeatmapvisualiser()
        output_path = tmp_path / "shap_result.png"
        visualiser.save(attributions, output_path, inputs=sample_images)
        assert output_path.exists()

    def test_tabular_workflow(self, simple_mlp, sample_tabular, tmp_path):
        """Test tabular data workflow"""
        visualiser = TabularBarChartvisualiser(feature_names=[f"feature_{i}" for i in range(10)])
        attributions = torch.randn_like(sample_tabular)
        output_path = tmp_path / "tabular_result.png"
        visualiser.save(attributions, output_path)
        assert output_path.exists()

    def test_config_driven_workflow(self, simple_cnn, sample_images):
        """Test config-driven workflow via method_from_config + create_explainer"""
        config = SimpleNamespace(framework="captum", algorithm="Saliency")
        method = method_from_config(config)
        explainer = create_explainer(method)
        assert explainer is not None


class TestExplainFunction:
    """Test the top-level explain() API"""

    @pytest.mark.skipif(
        not pytest.importorskip("captum", reason="Captum not installed"),
        reason="Captum not available",
    )
    def test_explain_captum_image(self, simple_cnn, sample_images, tmp_path):
        """explain() with captum + image visualiser"""
        config = SimpleNamespace(
            experiment_name="test",
            transparency=SimpleNamespace(
                framework="captum",
                algorithm="IntegratedGradients",
                visualisers=["image"],
                output_dir=str(tmp_path),
            ),
        )
        result = explain(config, simple_cnn, sample_images, target=0)

        assert "attributions" in result
        assert "visualisations" in result
        assert "run_dir" in result
        assert isinstance(result["attributions"], torch.Tensor)
        assert "image" in result["visualisations"]
        run_dir = result["run_dir"]
        assert (run_dir / "attributions.pt").exists()
        assert (run_dir / "image.png").exists()
        assert (run_dir / "metadata.json").exists()

    def test_explain_incompatible_visualiser_raises(self, simple_cnn, sample_images, tmp_path):
        """explain() raises VisualiserIncompatibilityError for SHAP image + KernelExplainer"""
        from raitap.transparency.methods_registry import VisualiserIncompatibilityError

        config = SimpleNamespace(
            experiment_name="test",
            transparency=SimpleNamespace(
                framework="shap",
                algorithm="KernelExplainer",
                visualisers=["image"],
                output_dir=str(tmp_path),
            ),
        )
        with pytest.raises(VisualiserIncompatibilityError):
            explain(config, simple_cnn, sample_images)


class TestEndToEndWorkflows:
    """Test complete workflows"""

    @pytest.mark.skipif(
        not pytest.importorskip("captum", reason="Captum not installed"),
        reason="Captum not available",
    )
    def test_end_to_end_captum(self, simple_cnn, sample_images, tmp_path):
        """Full workflow: create -> compute attributions -> visualize -> save"""
        explainer = create_explainer(Captum.IntegratedGradients)
        attributions = explainer.compute_attributions(simple_cnn, sample_images, target=0)
        assert isinstance(attributions, torch.Tensor)

        visualiser = ImageHeatmapvisualiser()
        output_path = tmp_path / "result.png"
        visualiser.save(attributions, output_path, inputs=sample_images)
        assert output_path.exists()

    @pytest.mark.skipif(
        not pytest.importorskip("shap", reason="SHAP not installed"), reason="SHAP not available"
    )
    def test_end_to_end_shap(self, simple_cnn, sample_images, tmp_path):
        """Full workflow with SHAP"""
        explainer = create_explainer(SHAP.GradientExplainer)
        background = sample_images[:2]
        attributions = explainer.compute_attributions(
            simple_cnn, sample_images, background_data=background, target=0
        )
        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape == sample_images.shape

        visualiser = ImageHeatmapvisualiser()
        output_path = tmp_path / "shap_result.png"
        visualiser.save(attributions, output_path, inputs=sample_images)
        assert output_path.exists()

    def test_tabular_workflow(self, simple_mlp, sample_tabular, tmp_path):
        """Test tabular data workflow"""
        # This test doesn't require actual attribution computation
        # Just test the visualiser works
        visualiser = TabularBarChartvisualiser(feature_names=[f"feature_{i}" for i in range(10)])

        # Mock attributions
        attributions = torch.randn_like(sample_tabular)

        output_path = tmp_path / "tabular_result.png"
        visualiser.save(attributions, output_path)

        assert output_path.exists()

    def test_config_driven_workflow(self, simple_cnn, sample_images):
        """Test Hydra-style config workflow"""
        # Simulate config values
        config = SimpleNamespace(framework="captum", algorithm="Saliency")

        # Translate config → registry
        method = method_from_config(config)

        # Create explainer
        explainer = create_explainer(method, modality="image")

        # This should work without errors
        assert explainer is not None
        # Algorithm is correctly set internally (implementation detail)
