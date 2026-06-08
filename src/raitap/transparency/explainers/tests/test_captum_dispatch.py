"""Captum dispatch via the per-entry invoker seam (#266, #269)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from raitap.transparency.explainers.captum_explainer import (
    CaptumExplainer,
    _default_captum_invoker,
)

if TYPE_CHECKING:
    import torch


def test_default_invoker_is_importable_callable() -> None:
    assert callable(_default_captum_invoker)


def test_saliency_entry_has_no_custom_invoker() -> None:
    # Default Captum methods carry invoker=None -> routed to _default_captum_invoker.
    assert CaptumExplainer.algorithm_registry["Saliency"].invoker is None


@pytest.mark.usefixtures("needs_captum")
def test_custom_invoker_overrides_default(
    simple_cnn: torch.nn.Module, sample_images: torch.Tensor
) -> None:
    explainer = CaptumExplainer("Saliency")
    sentinel = object()

    def fake_invoker(ctx):  # noqa: ANN001, ANN202
        return sentinel

    hints = CaptumExplainer.algorithm_registry["Saliency"]
    original = hints.invoker
    object.__setattr__(hints, "invoker", fake_invoker)  # frozen dataclass
    try:
        result = explainer.compute_attributions(simple_cnn, sample_images, target=0)
        assert result is sentinel
    finally:
        object.__setattr__(hints, "invoker", original)
