from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from raitap.transparency.contracts import MethodFamily
from raitap.transparency.visualisers.image_rendering import (
    IMAGE_RENDERER_REGISTRY,
    RaitapHouseRenderer,
    image_renderer,
    resolve_image_renderer,
)


def test_house_renderer_signed_returns_mappable() -> None:
    fig, ax = plt.subplots()
    attr = np.linspace(-1.0, 1.0, 3 * 4 * 4).reshape(3, 4, 4).astype(np.float32)
    image = np.zeros((4, 4, 3), dtype=np.float32)
    mappable = RaitapHouseRenderer().draw(ax, attr, image, sign="all")
    assert mappable is not None
    vmin, vmax = mappable.get_clim()
    assert vmin == -vmax
    plt.close(fig)


def test_house_renderer_positive_is_non_negative_floor() -> None:
    fig, ax = plt.subplots()
    attr = np.abs(np.linspace(0.0, 1.0, 16)).reshape(1, 4, 4).astype(np.float32)
    mappable = RaitapHouseRenderer().draw(ax, attr, None, sign="positive")
    assert mappable is not None
    vmin, vmax = mappable.get_clim()
    assert vmin == 0.0 and vmax > 0.0
    plt.close(fig)


def test_unknown_library_resolves_to_house() -> None:
    renderer, sign = resolve_image_renderer("does-not-exist", frozenset())
    assert isinstance(renderer, RaitapHouseRenderer)
    assert sign == "all"


def test_cam_family_resolves_to_positive_sign() -> None:
    _, sign = resolve_image_renderer(None, frozenset({MethodFamily.CAM}))
    assert sign == "positive"


def test_decorator_registers_by_library() -> None:
    @image_renderer(for_library="unittest-lib")
    class _Dummy:
        def draw(
            self, ax: object, attr: object, image: object, *, sign: str = "all", **style: object
        ) -> None:
            return None

    assert isinstance(IMAGE_RENDERER_REGISTRY["unittest-lib"], _Dummy)
    renderer, _ = resolve_image_renderer("unittest-lib", frozenset())
    assert isinstance(renderer, _Dummy)
    del IMAGE_RENDERER_REGISTRY["unittest-lib"]


def test_shap_native_matches_manual_recipe() -> None:
    pytest.importorskip("shap")
    import matplotlib.pyplot as plt

    from raitap.transparency.visualisers.image_rendering import ShapNativeRenderer
    from raitap.transparency.visualisers.shap_visualisers import (
        _image_heatmap,
        _symmetric_vmin_vmax,
    )

    shap_hwc = np.linspace(-2, 2, 4 * 4 * 3).reshape(4, 4, 3).astype(np.float32)
    fig, ax = plt.subplots()
    im = ShapNativeRenderer().draw(ax, shap_hwc, None)
    assert im is not None
    expected = _image_heatmap(shap_hwc)
    vmin, vmax = _symmetric_vmin_vmax(expected, 99.9)
    assert im.get_clim() == (vmin, vmax)
    drawn = im.get_array()
    assert drawn is not None
    np.testing.assert_allclose(np.asarray(drawn), expected)
    plt.close(fig)


def test_captum_native_draws_and_returns_mappable() -> None:
    pytest.importorskip("captum")
    import matplotlib.pyplot as plt

    from raitap.transparency.visualisers.image_rendering import CaptumNativeRenderer

    attr = np.linspace(-1, 1, 8 * 8 * 3).reshape(8, 8, 3).astype(np.float32)
    image = np.abs(attr)
    fig, ax = plt.subplots()
    im = CaptumNativeRenderer().draw(ax, attr, image, sign="all", method="heat_map")
    assert im is not None
    plt.close(fig)


def test_captum_native_flat_for_degenerate() -> None:
    pytest.importorskip("captum")
    import matplotlib.pyplot as plt

    from raitap.transparency.visualisers.image_rendering import CaptumNativeRenderer

    attr = np.zeros((8, 8, 3), dtype=np.float32)
    fig, ax = plt.subplots()
    im = CaptumNativeRenderer().draw(ax, attr, np.zeros_like(attr), sign="all")
    assert im is None  # flat-rendered, no mappable; must not raise
    plt.close(fig)


def test_builtin_libraries_registered() -> None:
    from raitap.transparency.visualisers.image_rendering import (
        IMAGE_RENDERER_REGISTRY,
        CaptumNativeRenderer,
        ShapNativeRenderer,
    )

    assert isinstance(IMAGE_RENDERER_REGISTRY["shap"], ShapNativeRenderer)
    assert isinstance(IMAGE_RENDERER_REGISTRY["captum"], CaptumNativeRenderer)


def test_resolver_shap_vs_captum_shapley_trap() -> None:
    from raitap.transparency.contracts import MethodFamily
    from raitap.transparency.visualisers.image_rendering import (
        CaptumNativeRenderer,
        ShapNativeRenderer,
    )

    r_shap, _ = resolve_image_renderer("shap", frozenset({MethodFamily.SHAPLEY}))
    r_cap, sign = resolve_image_renderer(
        "captum", frozenset({MethodFamily.SHAPLEY, MethodFamily.PERTURBATION})
    )
    assert isinstance(r_shap, ShapNativeRenderer)
    assert isinstance(r_cap, CaptumNativeRenderer) and sign == "all"
    _, cam_sign = resolve_image_renderer(
        "captum", frozenset({MethodFamily.GRADIENT, MethodFamily.CAM})
    )
    assert cam_sign == "positive"


def test_image_renderer_on_public_adapters_facade() -> None:
    from raitap import adapters

    assert hasattr(adapters, "image_renderer")

    @adapters.image_renderer(for_library="superxai-test")
    class _SuperXAIRenderer:
        def draw(
            self, ax: object, attr: object, image: object, *, sign: str = "all", **style: object
        ) -> None:
            return None

    from raitap.transparency.visualisers.image_rendering import resolve_image_renderer

    renderer, _ = resolve_image_renderer("superxai-test", frozenset())
    assert isinstance(renderer, _SuperXAIRenderer)
    # cleanup global registry to avoid cross-test leakage
    from raitap.transparency.visualisers.image_rendering import IMAGE_RENDERER_REGISTRY

    del IMAGE_RENDERER_REGISTRY["superxai-test"]
