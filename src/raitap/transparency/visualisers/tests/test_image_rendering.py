from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from raitap.transparency.contracts import MethodFamily
from raitap.transparency.visualisers.image_rendering import (
    IMAGE_RENDERER_REGISTRY,
    RaitapHouseRenderer,
    image_renderer,
    resolve_image_renderer,
)


def test_house_renderer_signed_returns_mappable():
    fig, ax = plt.subplots()
    attr = np.linspace(-1.0, 1.0, 3 * 4 * 4).reshape(3, 4, 4).astype(np.float32)
    image = np.zeros((4, 4, 3), dtype=np.float32)
    mappable = RaitapHouseRenderer().draw(ax, attr, image, sign="all")
    assert mappable is not None
    vmin, vmax = mappable.get_clim()
    assert vmin == -vmax
    plt.close(fig)


def test_house_renderer_positive_is_non_negative_floor():
    fig, ax = plt.subplots()
    attr = np.abs(np.linspace(0.0, 1.0, 16)).reshape(1, 4, 4).astype(np.float32)
    mappable = RaitapHouseRenderer().draw(ax, attr, None, sign="positive")
    vmin, vmax = mappable.get_clim()
    assert vmin == 0.0 and vmax > 0.0
    plt.close(fig)


def test_unknown_library_resolves_to_house():
    renderer, sign = resolve_image_renderer("does-not-exist", frozenset())
    assert isinstance(renderer, RaitapHouseRenderer)
    assert sign == "all"


def test_cam_family_resolves_to_positive_sign():
    _, sign = resolve_image_renderer(None, frozenset({MethodFamily.CAM}))
    assert sign == "positive"


def test_decorator_registers_by_library():
    @image_renderer(for_library="unittest-lib")
    class _Dummy:
        def draw(self, ax, attr, image, *, sign="all", **style):
            return None

    assert isinstance(IMAGE_RENDERER_REGISTRY["unittest-lib"], _Dummy)
    renderer, _ = resolve_image_renderer("unittest-lib", frozenset())
    assert isinstance(renderer, _Dummy)
    del IMAGE_RENDERER_REGISTRY["unittest-lib"]
