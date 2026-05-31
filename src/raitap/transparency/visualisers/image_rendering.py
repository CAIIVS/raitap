"""Shared per-Axes attribution renderers.

A *renderer* paints ONE attribution map onto ONE existing matplotlib Axes. It
owns no figure, no layout, and is not user-selected — visualisers (which own the
figure/layout) delegate the inner paint to a renderer. Renderers are chosen
automatically from explainer provenance via :func:`resolve_image_renderer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable


class ImageAttributionRenderer(Protocol):
    """Paint one attribution onto ``ax``; return the mappable (for a colorbar)."""

    def draw(
        self,
        ax: Axes,
        attr: np.ndarray,
        image: np.ndarray | None,
        *,
        sign: str = "all",
        **style: Any,
    ) -> ScalarMappable | None: ...


def _signed_channel_sum(attr: np.ndarray) -> np.ndarray:
    """Collapse ``(C, H, W)`` / ``(H, W, C)`` / ``(H, W)`` to a 2-D signed map."""
    arr = np.asarray(attr, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        return arr[..., 0] if arr.shape[-1] == 1 else arr.sum(axis=-1)
    return arr


def _symmetric_clim(values: np.ndarray, outlier_perc: float = 99.9) -> tuple[float, float]:
    abs_vals = np.abs(values)
    if abs_vals.size == 0:
        return -1.0, 1.0
    max_val = float(np.nanpercentile(abs_vals, outlier_perc))
    if not np.isfinite(max_val) or max_val == 0.0:
        return -1.0, 1.0
    return -max_val, max_val


def _grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[-1] == 3:
        return 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    return image.mean(axis=-1)


class RaitapHouseRenderer:
    """Dependency-free default renderer (no shap/captum import).

    ``sign="all"`` -> signed diverging (``bwr``, symmetric 99.9-percentile scale).
    ``sign="positive"`` -> non-negative sequential (``inferno``, ``[0, p99.9]``).
    Grayscale background of ``image`` at alpha 0.15 when provided.
    """

    def draw(self, ax: Axes, attr: np.ndarray, image: np.ndarray | None, *, sign: str = "all", **style: Any) -> ScalarMappable | None:
        heat = _signed_channel_sum(attr)
        if image is not None:
            ax.imshow(_grayscale(image), cmap="gray", alpha=0.15)
        if sign == "positive":
            heat = np.abs(heat)
            vmax = float(np.nanpercentile(heat, 99.9)) if heat.size else 1.0
            vmax = vmax if (np.isfinite(vmax) and vmax > 0) else 1.0
            im = ax.imshow(heat, cmap="inferno", vmin=0.0, vmax=vmax)
        else:
            vmin, vmax = _symmetric_clim(heat)
            im = ax.imshow(heat, cmap="bwr", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        return im
