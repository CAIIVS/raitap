"""Tests for DetectionImageVisualiser."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
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
from raitap.transparency.visualisers.image_rendering import RaitapHouseRenderer
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

    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name.startswith("raitap.transparency.visualisers")
    }
    for name in saved:
        del sys.modules[name]

    try:
        pkg = importlib.import_module("raitap.transparency.visualisers")
        assert hasattr(pkg, "DetectionImageVisualiser")
        assert pkg.DetectionImageVisualiser is DetectionImageVisualiser or (
            pkg.DetectionImageVisualiser.__name__ == "DetectionImageVisualiser"
        )
    finally:
        # Restore the original module instances. Re-importing the package builds
        # fresh module objects (and a fresh image-renderer registry); leaving
        # those in sys.modules would desync the singleton registry from
        # references cached at import time elsewhere (e.g. raitap.adapters),
        # breaking later tests that rely on the registry being a singleton.
        sys.modules.update(saved)
        # importlib.import_module rebinds parent-package attributes (e.g.
        # raitap.transparency.visualisers) to the freshly-created modules.
        # Restoring sys.modules alone leaves those dangling, so attribute-walk
        # resolution (monkeypatch by dotted path) and sys.modules resolution
        # (a lazy `from ... import`) disagree. Re-bind each saved module onto
        # its parent to fully restore the prior state.
        for name, module in saved.items():
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and child:
                setattr(parent, child, module)


def test_visualiser_upsamples_low_res_cam_to_full_image_extent() -> None:
    """LayerGradCam yields a low-res spatial map; the overlay must span the
    whole image, not sit in a top-left corner patch. Regression guard for the
    detection visualiser missing the bilinear-upsample the classification path
    already has (issue #203)."""
    vis = DetectionImageVisualiser()
    cam = torch.rand(1, 1, 8, 8)  # low-res CAM map, e.g. from LayerGradCam
    inputs = torch.rand(1, 3, 64, 64)
    ctx = VisualisationContext(
        algorithm="LayerGradCam",
        sample_names=["img_0"],
        show_sample_names=False,
        detection_box=_box(label_name="car"),
    )

    fig = vis.visualise(cam, inputs, context=ctx)
    try:
        main_ax = fig.get_axes()[0]
        overlay = main_ax.images[1]  # images[0] = original; images[1] = attribution overlay
        left, right, bottom, top = overlay.get_extent()
        assert abs(right - left) == pytest.approx(64.0, abs=2.0)
        assert abs(top - bottom) == pytest.approx(64.0, abs=2.0)
    finally:
        plt.close(fig)


def test_init_stores_render_style_fields() -> None:
    vis = DetectionImageVisualiser(
        method="heat_map",
        sign="positive",
        show_colorbar=True,
        title="Integrated Gradients",
    )
    assert vis.method == "heat_map"
    assert vis.sign == "positive"
    assert vis.show_colorbar is True
    assert vis.title == "Integrated Gradients"


def test_init_defaults_are_none_sentinels() -> None:
    vis = DetectionImageVisualiser()
    assert vis.method is None
    assert vis.sign is None
    assert vis.show_colorbar is None
    assert vis.title is None


class _SpyRenderer:
    """Records the sign + style kwargs forwarded by the visualiser."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def draw(self, ax, attr, image, *, sign="all", **style):
        self.calls.append((sign, dict(style)))
        ax.imshow(image if image is not None else attr)
        return None


def test_visualise_forwards_set_style_into_renderer_draw(monkeypatch) -> None:
    spy = _SpyRenderer()
    monkeypatch.setattr(
        "raitap.transparency.visualisers.image_rendering.resolve_image_renderer",
        lambda source_library, method_families: (spy, "all"),
    )
    vis = DetectionImageVisualiser(method="heat_map", show_colorbar=True)
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    ctx = VisualisationContext(
        algorithm="x",
        sample_names=None,
        show_sample_names=False,
        detection_box=_box(label_name="car"),
    )

    vis.visualise(attributions, inputs, context=ctx)

    assert len(spy.calls) == 1
    _, style = spy.calls[0]
    assert style == {"method": "heat_map", "show_colorbar": True}


def test_visualise_forwards_no_style_when_fields_unset(monkeypatch) -> None:
    spy = _SpyRenderer()
    monkeypatch.setattr(
        "raitap.transparency.visualisers.image_rendering.resolve_image_renderer",
        lambda source_library, method_families: (spy, "all"),
    )
    vis = DetectionImageVisualiser()
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    ctx = VisualisationContext(
        algorithm="x",
        sample_names=None,
        show_sample_names=False,
        detection_box=_box(label_name="car"),
    )

    vis.visualise(attributions, inputs, context=ctx)

    assert len(spy.calls) == 1
    sign, style = spy.calls[0]
    assert style == {}
    assert sign == "all"


def _ctx() -> VisualisationContext:
    return VisualisationContext(
        algorithm="x",
        sample_names=None,
        show_sample_names=False,
        detection_box=_box(label_name="car"),
    )


def test_warns_when_method_set_but_renderer_ignores_method(monkeypatch) -> None:
    monkeypatch.setattr(
        "raitap.transparency.visualisers.image_rendering.resolve_image_renderer",
        lambda source_library, method_families: (RaitapHouseRenderer(), "all"),
    )
    vis = DetectionImageVisualiser(method="heat_map")
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    with pytest.warns(UserWarning, match="method"):
        vis.visualise(attributions, inputs, context=_ctx())


def test_warns_when_sign_set_but_renderer_cannot_honour_it(monkeypatch) -> None:
    monkeypatch.setattr(
        "raitap.transparency.visualisers.image_rendering.resolve_image_renderer",
        lambda source_library, method_families: (RaitapHouseRenderer(), "all"),
    )
    vis = DetectionImageVisualiser(sign="negative")
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    with pytest.warns(UserWarning, match="sign"):
        vis.visualise(attributions, inputs, context=_ctx())


def test_no_warning_when_sign_is_honoured(monkeypatch) -> None:
    monkeypatch.setattr(
        "raitap.transparency.visualisers.image_rendering.resolve_image_renderer",
        lambda source_library, method_families: (RaitapHouseRenderer(), "all"),
    )
    vis = DetectionImageVisualiser(sign="positive")
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vis.visualise(attributions, inputs, context=_ctx())


def test_no_warning_when_renderer_has_no_capability_metadata(monkeypatch) -> None:
    spy = _SpyRenderer()  # no capability attrs -> assume honours all
    monkeypatch.setattr(
        "raitap.transparency.visualisers.image_rendering.resolve_image_renderer",
        lambda source_library, method_families: (spy, "all"),
    )
    vis = DetectionImageVisualiser(method="heat_map", sign="negative")
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vis.visualise(attributions, inputs, context=_ctx())


def test_config_sign_overrides_family_auto(monkeypatch) -> None:
    spy = _SpyRenderer()
    # family-auto would yield "positive", but config sign must win.
    monkeypatch.setattr(
        "raitap.transparency.visualisers.image_rendering.resolve_image_renderer",
        lambda source_library, method_families: (spy, "positive"),
    )
    vis = DetectionImageVisualiser(sign="negative")
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    vis.visualise(attributions, inputs, context=_ctx())
    sign, _ = spy.calls[0]
    assert sign == "negative"


def test_family_auto_sign_used_when_config_sign_unset(monkeypatch) -> None:
    spy = _SpyRenderer()
    monkeypatch.setattr(
        "raitap.transparency.visualisers.image_rendering.resolve_image_renderer",
        lambda source_library, method_families: (spy, "positive"),
    )
    vis = DetectionImageVisualiser()  # sign unset
    inputs = torch.rand(1, 3, 32, 32)
    attributions = torch.zeros_like(inputs)
    vis.visualise(attributions, inputs, context=_ctx())
    sign, _ = spy.calls[0]
    assert sign == "positive"


def test_title_surfaces_as_report_group_name() -> None:
    from raitap.reporting.builder import _visualiser_group_name

    titled = DetectionImageVisualiser(title="Integrated Gradients")
    assert _visualiser_group_name(titled, 0) == "Integrated Gradients"


def test_untitled_visualiser_falls_back_to_class_index_group_name() -> None:
    from raitap.reporting.builder import _visualiser_group_name

    untitled = DetectionImageVisualiser()
    assert _visualiser_group_name(untitled, 2) == "DetectionImageVisualiser_2"
