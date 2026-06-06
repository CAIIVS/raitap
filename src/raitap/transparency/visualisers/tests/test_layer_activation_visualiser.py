"""Tests for the layer-activation visualiser (#267)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from raitap.transparency.contracts import ExplanationOutputSpace
from raitap.transparency.visualisers.layer_activation_visualiser import (
    LayerActivationVisualiser,
)


def test_compat_declares_layer_activation_only() -> None:
    assert LayerActivationVisualiser.supported_output_spaces == frozenset(
        {ExplanationOutputSpace.LAYER_ACTIVATION}
    )


def test_visualise_conv_layer_one_bar_per_channel() -> None:
    fig = LayerActivationVisualiser().visualise(torch.randn(2, 8, 4, 4))  # (B, C, H, W)
    # 2 samples -> 2 axes, each a bar per channel (8).
    assert len(fig.axes) == 2
    assert all(len(ax.patches) == 8 for ax in fig.axes)
    plt.close(fig)


def test_visualise_linear_layer_one_bar_per_feature() -> None:
    fig = LayerActivationVisualiser().visualise(torch.randn(2, 16))  # (B, F)
    assert len(fig.axes) == 2
    assert all(len(ax.patches) == 16 for ax in fig.axes)
    plt.close(fig)


def test_visualise_sequence_layer_renders_heatmap() -> None:
    # (B, tokens, hidden) e.g. a ViT block: heatmap path, not flattened bars.
    fig = LayerActivationVisualiser().visualise(torch.randn(1, 10, 32))
    sample_axes = [ax for ax in fig.axes if ax.images]
    assert len(sample_axes) == 1  # one imshow heatmap, no bars
    assert all(len(ax.patches) == 0 for ax in sample_axes)
    plt.close(fig)


def test_visualise_caps_batch_at_max_samples() -> None:
    fig = LayerActivationVisualiser().visualise(torch.randn(20, 16), max_samples=4)
    assert len(fig.axes) == 4
    plt.close(fig)
