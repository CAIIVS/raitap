"""Shared per-Axes attribution renderers.

A *renderer* paints ONE attribution map onto ONE existing matplotlib Axes. It
owns no figure, no layout, and is not user-selected — visualisers (which own the
figure/layout) delegate the inner paint to a renderer. Renderers are chosen
automatically from explainer provenance via :func:`resolve_image_renderer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from raitap.transparency.contracts import MethodFamily

if TYPE_CHECKING:
    from collections.abc import Callable

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
    ) -> ScalarMappable | None:
        """Render ``attr`` onto ``ax``; return the drawn mappable (or ``None``)."""


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

    def draw(
        self,
        ax: Axes,
        attr: np.ndarray,
        image: np.ndarray | None,
        *,
        sign: str = "all",
        **style: Any,
    ) -> ScalarMappable | None:
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


IMAGE_RENDERER_REGISTRY: dict[str, ImageAttributionRenderer] = {}


def image_renderer(*, for_library: str) -> Callable[[type], type]:
    """Register a renderer instance under a library name (``source_library``).

    Public surface for plugin authors. The renderer auto-applies to any explainer
    whose ``registry_name`` equals ``for_library``.
    """

    def wrap(cls: type) -> type:
        IMAGE_RENDERER_REGISTRY[for_library] = cls()
        return cls

    return wrap


def resolve_image_renderer(
    source_library: str | None,
    method_families: frozenset[MethodFamily],
) -> tuple[ImageAttributionRenderer, str]:
    """Pick (renderer, sign) from provenance. Unknown/None library -> house."""
    renderer = IMAGE_RENDERER_REGISTRY.get(source_library or "", RaitapHouseRenderer())
    sign = "positive" if MethodFamily.CAM in method_families else "all"
    return renderer, sign


class ShapNativeRenderer:
    """SHAP-native recipe (red_transparent_blue, grayscale bg, +/-99.9 percentile).

    Mirrors ``shap.plots.image``. ``attr`` is channels-last (H,W,C) or (H,W);
    ``image`` is the normalised (H,W,C) original or None. ``sign`` is ignored —
    SHAP attributions are always signed-diverging.
    """

    def draw(
        self,
        ax: Axes,
        attr: np.ndarray,
        image: np.ndarray | None,
        *,
        sign: str = "all",
        **style: Any,
    ) -> ScalarMappable:
        from raitap.transparency.visualisers.shap_visualisers import (
            _image_heatmap,
            _red_transparent_blue,
            _rgb_to_grayscale,
            _symmetric_vmin_vmax,
        )

        cmap = style.get("cmap") or _red_transparent_blue()
        overlay_alpha = float(style.get("overlay_alpha", 0.15))
        outlier_perc = float(style.get("outlier_perc", 99.9))
        heatmap = _image_heatmap(np.asarray(attr, dtype=np.float32))
        vmin, vmax = _symmetric_vmin_vmax(heatmap, outlier_perc)
        if image is not None:
            ax.imshow(_rgb_to_grayscale(image), cmap="gray", alpha=overlay_alpha)
        return ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)


class CaptumNativeRenderer:
    """Captum-native recipe via ``captum.attr.visualization.visualize_image_attr``.

    ``attr`` channels-last (H,W,C); ``image`` normalised (H,W,C). ``method``
    (default ``"blended_heat_map"``) and other Captum styling are forwarded via
    ``**style``. Returns the drawn mappable, or ``None`` when the slice is a valid
    all-zero map (rendered flat instead of crashing — see #206/#207).
    """

    def draw(
        self,
        ax: Axes,
        attr: np.ndarray,
        image: np.ndarray | None,
        *,
        sign: str = "all",
        **style: Any,
    ) -> ScalarMappable | None:
        from captum.attr import visualization as viz
        from matplotlib.figure import Figure

        from raitap.transparency.visualisers.captum_visualisers import (
            _captum_normalisation_degenerate,
            _last_mappable,
            _render_flat_attribution,
        )

        method = style.pop("method", "blended_heat_map")
        title = style.pop("title", None)
        show_colorbar = bool(style.pop("show_colorbar", False))
        outlier_perc = float(style.get("outlier_perc", 2.0))
        if _captum_normalisation_degenerate(np.asarray(attr), sign, outlier_perc):
            _render_flat_attribution(ax, sign, title)
            return None
        # ``ax.figure`` is typed ``Figure | SubFigure``; visualisers always pass a
        # top-level ``Figure``'s axes, and visualize_image_attr's stub requires ``Figure``.
        fig = ax.figure
        assert isinstance(fig, Figure)
        viz.visualize_image_attr(
            attr,
            image,
            method=method,
            sign=sign,
            show_colorbar=show_colorbar,
            plt_fig_axis=(fig, ax),
            use_pyplot=False,
            **({"title": title} if title is not None else {}),
            **style,
        )
        return _last_mappable(ax)


IMAGE_RENDERER_REGISTRY["shap"] = ShapNativeRenderer()
IMAGE_RENDERER_REGISTRY["captum"] = CaptumNativeRenderer()
