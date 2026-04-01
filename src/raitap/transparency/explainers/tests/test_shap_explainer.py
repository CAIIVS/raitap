"""Tests for ShapExplainer implementation"""

from __future__ import annotations

import pytest
import torch

from raitap.transparency.explainers import ShapExplainer


class TestShapExplainer:
    """Test ShapExplainer wrapper"""

    def test_initialization(self) -> None:
        """Test explainer can be initialized"""
        explainer = ShapExplainer("GradientExplainer")
        assert explainer.algorithm == "GradientExplainer"

    @pytest.mark.usefixtures("needs_shap")
    def test_gradient_explainer_returns_tensor(
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor
    ) -> None:
        """Test SHAP GradientExplainer returns a tensor with the expected shape."""
        explainer = ShapExplainer("GradientExplainer")
        background = sample_images[:2]

        # With target: attributions have the same shape as the inputs
        attributions = explainer.compute_attributions(
            simple_cnn, sample_images, background_data=background, target=0
        )
        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape == sample_images.shape

        # Without target: SHAP returns one attribution map per output class
        # so there is an extra trailing class dimension
        all_class_attrs = explainer.compute_attributions(
            simple_cnn, sample_images, background_data=background
        )
        assert isinstance(all_class_attrs, torch.Tensor)
        assert all_class_attrs.shape[:-1] == sample_images.shape  # (..., num_classes)

    @pytest.mark.usefixtures("needs_shap")
    def test_no_background_falls_back_to_input(
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor
    ) -> None:
        """Without background_data, the explainer warns and uses the input as background."""
        explainer = ShapExplainer("GradientExplainer")
        attributions = explainer.compute_attributions(simple_cnn, sample_images, target=0)
        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape == sample_images.shape

    @pytest.mark.usefixtures("needs_shap")
    def test_invalid_algorithm_error(
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor
    ) -> None:
        """Invalid algorithm name raises ValueError with a helpful message."""
        explainer = ShapExplainer("NonExistentExplainer")

        with pytest.raises(ValueError) as exc_info:
            explainer.compute_attributions(simple_cnn, sample_images)

        assert "NonExistentExplainer" in str(exc_info.value)
        assert "shap" in str(exc_info.value)

    @pytest.mark.usefixtures("needs_shap")
    def test_gradient_explainer_with_base_batching(
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor
    ) -> None:
        """Mini-batching via BaseExplainer.explain works with per-sample targets."""
        explainer = ShapExplainer("GradientExplainer")
        background = sample_images[:2]

        result = explainer.explain(
            simple_cnn,
            sample_images,
            background_data=background,
            target=[0, 1, 2, 3],
            batch_size=2,
        )

        assert isinstance(result.attributions, torch.Tensor)
        assert result.attributions.shape == sample_images.shape

