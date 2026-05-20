"""Tests for DetectionImageVisualiser."""

from __future__ import annotations

import pytest
import torch

from raitap.transparency.contracts import (
    DetectionBox,
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    VisualisationContext,
)
from raitap.transparency.visualisers.detection_image_visualiser import (
    DetectionImageVisualiser,
)
from raitap.types import TaskKind


def _box(display_index: int = 0, raw_index: int = 0, label_name: str | None = None) -> DetectionBox:
    return DetectionBox(
        display_index=display_index,
        raw_index=raw_index,
        xyxy=(2.0, 3.0, 22.0, 23.0),
        score=0.87,
        label_index=4,
        label_name=label_name,
    )


def test_visualiser_supports_detection_output_space_only() -> None:
    assert DetectionImageVisualiser.supported_output_spaces == frozenset(
        {ExplanationOutputSpace.DETECTION_BOXES}
    )
    assert DetectionImageVisualiser.supported_payload_kinds == frozenset(
        {ExplanationPayloadKind.ATTRIBUTIONS}
    )
    assert DetectionImageVisualiser.supported_scopes == frozenset({ExplanationScope.LOCAL})
    assert DetectionImageVisualiser.supported_tasks == frozenset({TaskKind.detection})
    assert DetectionImageVisualiser.embeds_original_input is True


def test_visualiser_returns_figure_with_axis_limits_matching_image() -> None:
    vis = DetectionImageVisualiser()
    attributions = torch.zeros(1, 3, 64, 64)
    inputs = torch.rand(1, 3, 64, 64)
    ctx = VisualisationContext(
        algorithm="IntegratedGradients",
        sample_names=["img_0"],
        show_sample_names=False,
        detection_box=_box(label_name="car"),
    )

    fig = vis.visualise(attributions, inputs, context=ctx)
    main_ax = fig.get_axes()[0]
    xlim = main_ax.get_xlim()
    ylim = main_ax.get_ylim()
    assert xlim[1] - xlim[0] == pytest.approx(64.0, abs=2.0)
    assert abs(ylim[1] - ylim[0]) == pytest.approx(64.0, abs=2.0)


def test_visualiser_title_carries_label_name_and_score() -> None:
    vis = DetectionImageVisualiser()
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    ctx = VisualisationContext(
        algorithm="x",
        sample_names=None,
        show_sample_names=False,
        detection_box=_box(label_name="car"),
    )
    fig = vis.visualise(attributions, inputs, context=ctx)
    title = fig.get_axes()[0].get_title()
    assert "car" in title
    assert "0.87" in title


def test_visualiser_falls_back_to_class_id_when_label_name_missing() -> None:
    vis = DetectionImageVisualiser()
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    ctx = VisualisationContext(
        algorithm="x",
        sample_names=None,
        show_sample_names=False,
        detection_box=_box(label_name=None),
    )
    fig = vis.visualise(attributions, inputs, context=ctx)
    title = fig.get_axes()[0].get_title()
    assert "class 4" in title


def test_visualiser_raises_when_detection_box_missing() -> None:
    vis = DetectionImageVisualiser()
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    ctx = VisualisationContext(
        algorithm="x",
        sample_names=None,
        show_sample_names=False,
        detection_box=None,
    )
    with pytest.raises(ValueError, match="detection_box"):
        vis.visualise(attributions, inputs, context=ctx)


def test_detection_image_visualiser_is_importable_from_visualisers_package() -> None:
    """Fresh-process surface: importing raitap.transparency.visualisers must
    eagerly load detection_image_visualiser so the @visualisers.transparency
    side effect runs and the hydra-zen store knows about ``_target_:
    detection_image``."""
    import importlib
    import sys

    for mod in list(sys.modules):
        if mod.startswith("raitap.transparency.visualisers"):
            del sys.modules[mod]

    pkg = importlib.import_module("raitap.transparency.visualisers")
    assert hasattr(pkg, "DetectionImageVisualiser")
    assert pkg.DetectionImageVisualiser is DetectionImageVisualiser or (
        pkg.DetectionImageVisualiser.__name__ == "DetectionImageVisualiser"
    )
