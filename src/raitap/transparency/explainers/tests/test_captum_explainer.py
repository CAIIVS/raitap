"""Tests for CaptumExplainer implementation"""

from __future__ import annotations

import pytest
import torch

from raitap.transparency.explainers import CaptumExplainer


class TestCaptumExplainer:
    """Test CaptumExplainer wrapper"""

    def test_initialization(self) -> None:
        """Test explainer can be initialized"""
        explainer = CaptumExplainer("IntegratedGradients")
        assert explainer.algorithm == "IntegratedGradients"

    def test_integrated_gradients(
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor
    ) -> None:
        """Test IntegratedGradients computation"""
        explainer = CaptumExplainer("IntegratedGradients")
        attributions = explainer.compute_attributions(simple_cnn, sample_images, target=0)

        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape == sample_images.shape

    def test_saliency(self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor) -> None:
        """Test Saliency method"""
        explainer = CaptumExplainer("Saliency")
        attributions = explainer.compute_attributions(simple_cnn, sample_images, target=0)

        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape == sample_images.shape

    def test_batch_targets(self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor) -> None:
        """Test different target formats"""
        explainer = CaptumExplainer("Saliency")

        # Single int target
        attr1 = explainer.compute_attributions(simple_cnn, sample_images, target=0)
        assert attr1.shape == sample_images.shape

        # List of targets
        attr2 = explainer.compute_attributions(simple_cnn, sample_images, target=[0, 1, 2, 3])
        assert attr2.shape == sample_images.shape

        # Tensor targets
        attr3 = explainer.compute_attributions(
            simple_cnn, sample_images, target=torch.tensor([0, 1, 2, 3])
        )
        assert attr3.shape == sample_images.shape

    def test_invalid_method_error(
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor
    ) -> None:
        """Test error message for invalid method"""
        explainer = CaptumExplainer("NonExistentMethod")

        with pytest.raises(ValueError) as exc_info:
            explainer.compute_attributions(simple_cnn, sample_images, target=0)

        assert "NonExistentMethod" in str(exc_info.value)
        assert "captum.attr" in str(exc_info.value)
