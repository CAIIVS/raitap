"""Tests for CaptumExplainer implementation"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from raitap.transparency.explainers import CaptumExplainer

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.models.backend import OnnxBackend


class TestCaptumExplainer:
    """Test CaptumExplainer wrapper"""

    def test_initialization(self) -> None:
        """Test explainer can be initialized"""
        explainer = CaptumExplainer("IntegratedGradients")
        assert explainer.algorithm == "IntegratedGradients"

    @pytest.mark.usefixtures("needs_captum")
    def test_integrated_gradients(
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor
    ) -> None:
        """Test IntegratedGradients computation"""
        explainer = CaptumExplainer("IntegratedGradients")
        attributions = explainer.compute_attributions(simple_cnn, sample_images, target=0)

        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape == sample_images.shape

    @pytest.mark.usefixtures("needs_captum")
    def test_saliency(self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor) -> None:
        """Test Saliency method"""
        explainer = CaptumExplainer("Saliency")
        attributions = explainer.compute_attributions(simple_cnn, sample_images, target=0)

        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape == sample_images.shape

    @pytest.mark.usefixtures("needs_captum")
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

    @pytest.mark.usefixtures("needs_captum")
    def test_invalid_method_error(
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor
    ) -> None:
        """Test error message for invalid method"""
        explainer = CaptumExplainer("NonExistentMethod")

        with pytest.raises(ValueError) as exc_info:
            explainer.compute_attributions(simple_cnn, sample_images, target=0)

        assert "NonExistentMethod" in str(exc_info.value)
        assert "captum.attr" in str(exc_info.value)

    @pytest.mark.usefixtures("needs_captum")
    def test_layer_gradcam_with_layer_path(
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor
    ) -> None:
        """LayerGradCam resolves layer_path to a module before Captum init."""
        explainer = CaptumExplainer("LayerGradCam", layer_path="0")
        attributions = explainer.compute_attributions(simple_cnn, sample_images, target=0)

        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape[0] == sample_images.shape[0]

    @pytest.mark.usefixtures("needs_captum")
    def test_layer_gradcam_with_invalid_layer_path_raises(
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor
    ) -> None:
        """Invalid layer_path gives a clear error before Captum init."""
        explainer = CaptumExplainer("LayerGradCam", layer_path="does.not.exist")

        with pytest.raises(ValueError, match="Could not resolve layer_path"):
            explainer.compute_attributions(simple_cnn, sample_images, target=0)

    @pytest.mark.usefixtures("needs_captum")
    def test_occlusion_normalises_yaml_lists_to_tuples(
        self,
        monkeypatch: pytest.MonkeyPatch,
        simple_cnn: torch.nn.Module,
        sample_images: torch.Tensor,
    ) -> None:
        captured_kwargs: dict[str, object] = {}

        class _OcclusionStub:
            def __init__(self, model: torch.nn.Module, **kwargs: object) -> None:
                del model, kwargs

            def attribute(self, inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
                captured_kwargs.update(kwargs)
                return torch.zeros_like(inputs)

        monkeypatch.setattr("captum.attr.Occlusion", _OcclusionStub)

        explainer = CaptumExplainer("Occlusion")
        attributions = explainer.compute_attributions(
            simple_cnn,
            sample_images,
            target=0,
            sliding_window_shapes=[3, 4, 4],
            strides=[1, 2, 2],
        )

        assert isinstance(attributions, torch.Tensor)
        assert captured_kwargs["sliding_window_shapes"] == (3, 4, 4)
        assert captured_kwargs["strides"] == (1, 2, 2)

    @pytest.mark.usefixtures("needs_captum")
    def test_saliency_with_base_batching(
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor, tmp_path: Path
    ) -> None:
        """Mini-batching via BaseExplainer.explain works for Captum methods."""
        explainer = CaptumExplainer("Saliency")

        result = explainer.explain(
            simple_cnn,
            sample_images,
            run_dir=tmp_path / "transparency",
            target=[0, 1, 2, 3],
            batch_size=2,
        )

        assert isinstance(result.attributions, torch.Tensor)
        assert result.attributions.shape == sample_images.shape

    @pytest.mark.usefixtures("needs_captum", "needs_onnx")
    def test_feature_ablation_runs_with_onnx_backend(
        self,
        onnx_linear_backend: OnnxBackend,
        sample_tabular: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        explainer = CaptumExplainer("FeatureAblation")
        inputs = sample_tabular[:4]

        explainer.check_backend_compat(onnx_linear_backend)
        result = explainer.explain(
            onnx_linear_backend.as_model_for_explanation(),
            inputs,
            run_dir=tmp_path / "transparency",
            backend=onnx_linear_backend,
            target=0,
        )

        assert isinstance(result.attributions, torch.Tensor)
        assert result.attributions.shape == inputs.shape

    @pytest.mark.usefixtures("needs_captum", "needs_onnx")
    def test_feature_ablation_with_onnx_backend_supports_batched_explain(
        self,
        onnx_linear_backend: OnnxBackend,
        sample_tabular: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        explainer = CaptumExplainer("FeatureAblation")
        inputs = sample_tabular[:4]

        explainer.check_backend_compat(onnx_linear_backend)
        result = explainer.explain(
            onnx_linear_backend.as_model_for_explanation(),
            inputs,
            run_dir=tmp_path / "transparency",
            backend=onnx_linear_backend,
            target=0,
            batch_size=2,
        )

        assert isinstance(result.attributions, torch.Tensor)
        assert result.attributions.shape == inputs.shape
