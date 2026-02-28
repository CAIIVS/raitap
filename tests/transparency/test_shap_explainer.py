"""Tests for ShapExplainer implementation"""

from __future__ import annotations

import pytest
import torch

from raitap.transparency.explainers import ShapExplainer


class TestShapExplainer:
    """Test ShapExplainer wrapper"""

    def test_initialization(self):
        """Test explainer can be initialized"""
        explainer = ShapExplainer("GradientExplainer")
        assert explainer.algorithm == "GradientExplainer"

    @pytest.mark.skipif(
        not pytest.importorskip("shap", reason="SHAP not installed"), reason="SHAP not available"
    )
    def test_gradient_explainer_returns_tensor(self, simple_cnn, sample_images):
        """Test SHAP returns tensor"""
        explainer = ShapExplainer("GradientExplainer")
        background = sample_images[:2]

        attributions = explainer.compute_attributions(
            simple_cnn, sample_images, background_data=background
        )
        assert isinstance(attributions, torch.Tensor)

        # No background_data: warns and falls back to using the input as background
        attributions_no_bg = explainer.compute_attributions(simple_cnn, sample_images)
        assert isinstance(attributions_no_bg, torch.Tensor)
