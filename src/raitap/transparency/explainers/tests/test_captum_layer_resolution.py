"""Tests for captum ``Layer*`` resolution detection (#267)."""

from __future__ import annotations

import pytest

from raitap.transparency.explainers.captum_explainer import _needs_layer_resolution


@pytest.mark.parametrize(
    "algorithm",
    [
        "LayerGradCam",
        "GuidedGradCam",
        "LayerConductance",
        "LayerIntegratedGradients",
        "LayerActivation",
        "LayerDeepLift",
        "LayerGradientXActivation",
        "LayerLRP",
    ],
)
def test_layer_methods_need_resolution(algorithm: str) -> None:
    assert _needs_layer_resolution(algorithm) is True


@pytest.mark.parametrize(
    "algorithm",
    # NeuronConductance also takes ``layer`` but is out of scope (#269): the
    # predicate must NOT claim it.
    ["IntegratedGradients", "Saliency", "Occlusion", "KernelShap", "NeuronConductance"],
)
def test_non_layer_methods_do_not(algorithm: str) -> None:
    assert _needs_layer_resolution(algorithm) is False
