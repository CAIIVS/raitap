"""Tests for ShapExplainer implementation"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from raitap.transparency.explainers import ShapExplainer
from raitap.transparency.explainers.shap_explainer import _select_target_attributions

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.models.backend import OnnxBackend


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
        self, simple_cnn: torch.nn.Module, sample_images: torch.Tensor, tmp_path: Path
    ) -> None:
        """Mini-batching via AttributionOnlyExplainer.explain works with per-sample targets."""
        explainer = ShapExplainer("GradientExplainer")
        background = sample_images[:2]

        result = explainer.explain(
            simple_cnn,
            sample_images,
            run_dir=tmp_path / "transparency",
            background_data=background,
            target=[0, 1, 2, 3],
            raitap_kwargs={"batch_size": 2},
        )

        assert isinstance(result.attributions, torch.Tensor)
        assert result.attributions.shape == sample_images.shape

    def test_select_target_attributions_normalises_tensor_targets(self) -> None:
        shap_values = torch.tensor(
            [
                [[[[10.0, 11.0, 12.0]]]],
                [[[[20.0, 21.0, 22.0]]]],
            ]
        )

        selected = _select_target_attributions(
            shap_values,
            inputs_ndim=4,
            target=torch.tensor([2.0, 1.0]),
        )

        assert torch.equal(selected, torch.tensor([[[[12.0]]], [[[21.0]]]]))

    def test_compute_attributions_selects_per_sample_targets_with_cpu_shap_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_images: torch.Tensor,
    ) -> None:
        class FakeGradientExplainer:
            def __init__(
                self,
                model: torch.nn.Module,
                background_data: torch.Tensor,
                **_kwargs,
            ) -> None:
                self.model = model
                self.background_data = background_data

            def shap_values(self, inputs: torch.Tensor, **_kwargs) -> torch.Tensor:
                del _kwargs
                zeros = torch.zeros_like(inputs)
                class_maps = [zeros + class_index for class_index in range(3)]
                return torch.stack(class_maps, dim=-1)

        monkeypatch.setitem(
            sys.modules,
            "shap",
            SimpleNamespace(GradientExplainer=FakeGradientExplainer),
        )

        explainer = ShapExplainer("GradientExplainer")
        targets = torch.tensor([0.0, 1.0, 2.0, 1.0])

        attributions = explainer.compute_attributions(
            torch.nn.Identity(),
            sample_images,
            background_data=sample_images[:2],
            target=targets,
        )

        expected = torch.stack(
            [
                torch.zeros_like(sample_images[0]),
                torch.ones_like(sample_images[1]),
                torch.full_like(sample_images[2], 2.0),
                torch.ones_like(sample_images[3]),
            ]
        )
        assert torch.equal(attributions, expected)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for device-mismatch regression",
    )
    def test_select_target_attributions_accepts_cuda_targets_for_cpu_shap_values(self) -> None:
        shap_values = torch.tensor(
            [
                [[[[1.0, 2.0]]]],
                [[[[3.0, 4.0]]]],
            ]
        )
        target = torch.tensor([1, 0], device=torch.device("cuda"))

        selected = _select_target_attributions(
            shap_values,
            inputs_ndim=4,
            target=target,
        )

        assert torch.equal(selected, torch.tensor([[[[2.0]]], [[[3.0]]]]))

    @pytest.mark.usefixtures("needs_shap", "needs_onnx")
    def test_kernel_explainer_runs_with_onnx_backend(
        self,
        onnx_linear_backend: OnnxBackend,
        sample_tabular: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        explainer = ShapExplainer("KernelExplainer")
        inputs = sample_tabular[:4]
        background = sample_tabular[:2]

        explainer.check_backend_compat(onnx_linear_backend)
        result = explainer.explain(
            onnx_linear_backend.as_model_for_explanation(),
            inputs,
            run_dir=tmp_path / "transparency",
            backend=onnx_linear_backend,
            background_data=background,
            target=0,
            nsamples=10,
        )

        assert isinstance(result.attributions, torch.Tensor)
        assert result.attributions.shape == inputs.shape
