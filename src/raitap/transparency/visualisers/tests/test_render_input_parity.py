"""Tier-3 parity tests: assert the arrays raitap feeds into captum's image
visualizer, not the rendered pixels.

Verifies that:
- the ``original_image`` passed to ``visualize_image_attr`` is min-max
  normalised to [0, 1],
- the attribution array is not mutated before being handed off,
- low-res attributions are bilinearly upsampled to match image HxW before the
  call.

Patch target: ``captum.attr._utils.visualization.visualize_image_attr``
(``captum.attr.visualization`` resolves to the same module object).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch

from raitap.testing import make_tiny_classifier

pytestmark = [pytest.mark.e2e, pytest.mark.parity]


@pytest.fixture
def _captum() -> Any:
    """Skip the entire module when captum is not installed."""
    return pytest.importorskip("captum.attr")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_TARGET = "captum.attr._utils.visualization.visualize_image_attr"


# ---------------------------------------------------------------------------
# Test 1 - normalised image + unmutated attributions
# ---------------------------------------------------------------------------


def test_image_visualiser_feeds_normalised_image_and_unmutated_attr(
    _captum: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spy verifies original_image is [0,1]-normalised and attr equals computed
    values (no mutation)."""
    import matplotlib.pyplot as plt
    from captum.attr import IntegratedGradients

    from raitap.transparency.visualisers import CaptumImageVisualiser

    model = make_tiny_classifier(seed=0)
    torch.manual_seed(42)
    x = torch.randn(1, 3, 8, 8)
    attr_tensor = IntegratedGradients(model).attribute(x, target=0, n_steps=7)

    # Capture calls to visualize_image_attr
    captured: list[dict[str, Any]] = []

    def _spy(attr_arr: Any, original_image: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        captured.append({"attr": attr_arr, "original_image": original_image, "kwargs": kwargs})
        fig, ax = plt.subplots(1, 1)
        return fig, ax

    monkeypatch.setattr(_PATCH_TARGET, _spy)

    vis = CaptumImageVisualiser(method="heat_map", show_colorbar=False)
    fig = vis.visualise(attr_tensor, inputs=x, max_samples=1, include_original_input=False)
    plt.close(fig)

    assert len(captured) >= 1, "Spy never fired - patch target may be wrong"

    call = captured[0]

    # --- original_image must be min-max normalised to [0, 1] ---
    orig = call["original_image"]
    assert orig is not None, "original_image was None"
    assert float(orig.min()) >= -1e-6, "original_image min should be >= 0 (normalised)"
    assert float(orig.max()) <= 1.0 + 1e-6, "original_image max should be <= 1 (normalised)"

    # --- attr must equal the computed attribution values (no mutation) ---
    # Visualiser converts (1, C, H, W) tensor -> numpy -> transpose (C,H,W) -> (H,W,C)
    expected_hwc = np.transpose(attr_tensor.detach().cpu().numpy()[0], (1, 2, 0))
    captured_attr = call["attr"]
    assert captured_attr.shape == expected_hwc.shape, (
        f"attr shape mismatch: got {captured_attr.shape}, expected {expected_hwc.shape}"
    )
    np.testing.assert_allclose(
        captured_attr,
        expected_hwc,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Attr array handed to captum differs from computed attribution values",
    )


# ---------------------------------------------------------------------------
# Test 2 - low-res attributions are upsampled to image HxW
# ---------------------------------------------------------------------------


def test_low_res_attr_is_upsampled_to_image_hw(
    _captum: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LayerGradCam on the pool layer produces 1x1 attr; raitap must bilinearly
    upsample it to 16x16 before passing to visualize_image_attr.

    Note: model[0] (Conv2d, padding=1, stride=1) preserves spatial size, so
    model[2] (AdaptiveAvgPool2d(1)) is used - it reliably gives a 1x1 feature
    map that triggers the upsample path.
    """
    import matplotlib.pyplot as plt
    from captum.attr import LayerGradCam

    from raitap.transparency.visualisers import CaptumImageVisualiser

    model = make_tiny_classifier(seed=0)
    torch.manual_seed(0)
    x = torch.randn(1, 3, 16, 16)

    # model[2] is AdaptiveAvgPool2d(1) -> feature map is 1x1 -> GradCam gives (1,1,1,1)
    layer = model[2]
    attr_tensor = cast("torch.Tensor", LayerGradCam(model, layer).attribute(x, target=0))
    # Confirm attr is genuinely low-res (must be smaller than 16x16 to be meaningful)
    assert attr_tensor.shape[-2:] != (16, 16), (
        f"Expected low-res attr; got {attr_tensor.shape} - choose a different layer"
    )

    captured: list[dict[str, Any]] = []

    def _spy(attr_arr: Any, original_image: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        captured.append({"attr": attr_arr, "original_image": original_image})
        fig, ax = plt.subplots(1, 1)
        return fig, ax

    monkeypatch.setattr(_PATCH_TARGET, _spy)

    vis = CaptumImageVisualiser(method="heat_map", show_colorbar=False)
    fig = vis.visualise(attr_tensor, inputs=x, max_samples=1, include_original_input=False)
    plt.close(fig)

    assert len(captured) >= 1, "Spy never fired - patch target may be wrong"

    captured_attr = captured[0]["attr"]
    spatial = captured_attr.shape[:2]
    assert spatial == (16, 16), f"Expected attr spatial dims (16, 16) after upsample; got {spatial}"
