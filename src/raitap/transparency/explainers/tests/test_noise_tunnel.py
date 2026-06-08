"""NoiseTunnel (SmoothGrad/VarGrad) support (#269)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from raitap.transparency.contracts import MethodFamily
from raitap.transparency.explainers.captum_explainer import (
    CaptumExplainer,
    _noise_tunnel_base_choices,
    _noise_tunnel_invoker,
)
from raitap.types import Capability

if TYPE_CHECKING:
    from pathlib import Path

    import torch


def test_noise_tunnel_registered_as_stochastic_gradient() -> None:
    hints = CaptumExplainer.algorithm_registry["NoiseTunnel"]
    assert hints.stochastic is True
    assert MethodFamily.GRADIENT in hints.families
    assert Capability.AUTOGRAD in hints.requires
    assert hints.invoker is _noise_tunnel_invoker


def test_base_choices_are_non_layer_gradient_methods() -> None:
    choices = _noise_tunnel_base_choices()
    assert "Saliency" in choices
    assert "IntegratedGradients" in choices
    assert "LayerGradCam" not in choices
    assert "NoiseTunnel" not in choices


@pytest.mark.usefixtures("needs_captum")
def test_missing_base_algorithm_raises(
    simple_cnn: torch.nn.Module, sample_images: torch.Tensor
) -> None:
    explainer = CaptumExplainer("NoiseTunnel")
    with pytest.raises(ValueError, match="base_algorithm"):
        explainer.compute_attributions(simple_cnn, sample_images, target=0)


@pytest.mark.usefixtures("needs_captum")
def test_layer_base_rejected(simple_cnn: torch.nn.Module, sample_images: torch.Tensor) -> None:
    explainer = CaptumExplainer("NoiseTunnel", base_algorithm="LayerGradCam")
    with pytest.raises(ValueError, match="not supported"):
        explainer.compute_attributions(simple_cnn, sample_images, target=0)


@pytest.mark.usefixtures("needs_captum")
def test_unknown_base_rejected(simple_cnn: torch.nn.Module, sample_images: torch.Tensor) -> None:
    explainer = CaptumExplainer("NoiseTunnel", base_algorithm="DoesNotExist")
    with pytest.raises(ValueError, match="not supported"):
        explainer.compute_attributions(simple_cnn, sample_images, target=0)


@pytest.mark.usefixtures("needs_captum")
def test_smoothgrad_over_saliency_returns_input_shape(
    simple_cnn: torch.nn.Module, sample_images: torch.Tensor
) -> None:
    explainer = CaptumExplainer("NoiseTunnel", base_algorithm="Saliency")
    attributions = explainer.compute_attributions(
        simple_cnn, sample_images, target=0, nt_type="smoothgrad", nt_samples=2, stdevs=0.1
    )
    assert attributions.shape == sample_images.shape


@pytest.mark.usefixtures("needs_captum")
def test_explain_marks_result_stochastic(
    simple_cnn: torch.nn.Module, sample_images: torch.Tensor, tmp_path: Path
) -> None:
    explainer = CaptumExplainer("NoiseTunnel", base_algorithm="Saliency")
    result = explainer.explain(
        simple_cnn,
        sample_images,
        run_dir=str(tmp_path),
        target=0,
        nt_type="smoothgrad",
        nt_samples=2,
        raitap_kwargs={
            "input_metadata": {
                "kind": "image",
                "shape": tuple(sample_images.shape),
                "layout": "NCHW",
            },
        },
    )
    assert result.attributions.shape == sample_images.shape
    assert result.semantics.stochastic is True
