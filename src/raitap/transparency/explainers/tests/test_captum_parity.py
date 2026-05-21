# src/raitap/transparency/explainers/tests/test_captum_parity.py
"""Tier-1 parity: raitap's CaptumExplainer returns the same numbers as a direct
captum.attr call for the same config. Catches dropped kwargs, wrong target
wiring, or stray mutation in the data path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from raitap.testing import make_tiny_classifier

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.e2e, pytest.mark.parity]

_RTOL = 1e-5
_ATOL = 1e-6


@pytest.fixture
def _captum() -> Any:
    return pytest.importorskip("captum.attr")


def test_integrated_gradients_matches_direct_call(
    _captum: Any, seeded: Callable[..., None]
) -> None:
    seeded()
    model = make_tiny_classifier(seed=0)
    x = torch.randn(2, 3, 8, 8)
    baselines = torch.zeros_like(x)

    raw = _captum.IntegratedGradients(model).attribute(x, target=0, baselines=baselines, n_steps=7)
    from raitap.transparency.explainers import CaptumExplainer

    # n_steps is an .attribute() kwarg (not an __init__ kwarg) in captum's API;
    # pass it via attr_kwargs, not init_kwargs.
    out = CaptumExplainer("IntegratedGradients").compute_attributions(
        model, x, target=0, baselines=baselines, n_steps=7
    )
    assert torch.allclose(raw.detach().cpu(), out.detach().cpu(), rtol=_RTOL, atol=_ATOL)


def test_occlusion_matches_direct_call(_captum: Any, seeded: Callable[..., None]) -> None:
    seeded()
    model = make_tiny_classifier(seed=0)
    x = torch.randn(2, 3, 8, 8)

    raw = _captum.Occlusion(model).attribute(
        x, target=0, sliding_window_shapes=(3, 2, 2), strides=(1, 1, 1)
    )
    from raitap.transparency.explainers import CaptumExplainer

    out = CaptumExplainer("Occlusion").compute_attributions(
        model, x, target=0, sliding_window_shapes=[3, 2, 2], strides=[1, 1, 1]
    )
    assert torch.allclose(raw.detach().cpu(), out.detach().cpu(), rtol=_RTOL, atol=_ATOL)


def test_layer_gradcam_matches_direct_call(_captum: Any, seeded: Callable[..., None]) -> None:
    seeded()
    model = make_tiny_classifier(seed=0)
    x = torch.randn(2, 3, 8, 8)
    layer = model[0]

    raw = _captum.LayerGradCam(model, layer).attribute(x, target=0)
    from raitap.transparency.explainers import CaptumExplainer

    out = CaptumExplainer("LayerGradCam", layer=layer).compute_attributions(model, x, target=0)
    assert torch.allclose(raw.detach().cpu(), out.detach().cpu(), rtol=_RTOL, atol=_ATOL)
