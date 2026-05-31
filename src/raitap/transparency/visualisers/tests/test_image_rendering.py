from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from raitap.transparency.visualisers.image_rendering import RaitapHouseRenderer


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
