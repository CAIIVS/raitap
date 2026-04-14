"""SHAP visualisers for RAITAP.

Most visualisers wrap SHAP's plotting APIs (``shap.plots.*`` /
``shap.summary_plot``). ``ShapImageVisualiser`` is rendered manually with
Matplotlib so RAITAP can provide a consistent paired image/overlay layout,
titles, sample names, and colorbar handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from .base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor or array-like to numpy (float32)."""
    val: Any = x
    if hasattr(val, "detach"):
        val = val.detach()
    if hasattr(val, "cpu"):
        val = val.cpu()
    if hasattr(val, "numpy"):
        val = val.numpy()
    return np.asarray(val, dtype=np.float32)


def _close_and_return(fig: Figure) -> Figure:
    """Detach a figure from pyplot's interactive state then return it."""
    plt.close(fig)
    return fig


def _normalise_image(image: np.ndarray) -> np.ndarray:
    """Normalise an image array to the [0, 1] range when possible."""
    lo, hi = image.min(), image.max()
    if hi > lo:
        return (image - lo) / (hi - lo)
    return image


def _compose_title(base_title: str | None, sample_name: str | None = None) -> str | None:
    """Combine a base title and optional sample name.

    ``None`` means no base title is set and allows falling back to the sample
    name. An explicitly provided empty string is preserved as-is.
    """
    if base_title is None:
        return sample_name
    if base_title and sample_name:
        return f"{base_title}: {sample_name}"
    return base_title


def _resolve_title(
    *, explicit_title: str | None, fallback_title: str | None, sample_name: str | None = None
) -> str | None:
    """Resolve titles while preserving an explicitly provided empty string."""
    base_title = explicit_title if explicit_title is not None else fallback_title
    return _compose_title(base_title, sample_name)


def _display_image(ax: Any, image: np.ndarray) -> None:
    """Render an image on an axis with sensible defaults for grayscale/RGB input."""
    if image.ndim == 3 and image.shape[-1] == 1:
        ax.imshow(image[..., 0], cmap="gray")
        return
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)


def _default_image_title(algorithm: str | None) -> str:
    """Return the default SHAP attribution title for image plots."""
    return f"{algorithm} (SHAP)" if algorithm else "SHAP Image"


def _image_heatmap(values: np.ndarray) -> np.ndarray:
    """Reduce channel-wise SHAP values to a 2-D heatmap for display."""
    if values.ndim == 3:
        if values.shape[-1] == 1:
            return values[..., 0]
        return values.sum(axis=-1)
    return values


# ---------------------------------------------------------------------------
# Tabular / general visualisers
# Compatible with all SHAP explainer algorithms.
# ---------------------------------------------------------------------------


class ShapBarVisualiser(BaseVisualiser):
    """
    Mean absolute SHAP value bar chart via ``shap.summary_plot(plot_type='bar')``.

    Compatible with all SHAP explainer algorithms.
    """

    def __init__(self, feature_names: list[str] | None = None, max_display: int = 20):
        self.feature_names = feature_names
        self.max_display = max_display

    def visualise(
        self, attributions: torch.Tensor, inputs: torch.Tensor | None = None, **kwargs: Any
    ) -> Figure:
        """
        Args:
            attributions: ``(B, F)`` SHAP values tensor / array.
            inputs:       Original feature values ``(B, F)`` (used for colouring).
            **kwargs:     Forwarded to ``shap.summary_plot``.
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP visualiser is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

        values = _to_numpy(attributions)
        feats = _to_numpy(inputs) if inputs is not None else None

        fig = plt.figure()
        shap.summary_plot(
            values,
            features=feats,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=self.max_display,
            show=False,
            **kwargs,
        )
        return fig


class ShapBeeswarmVisualiser(BaseVisualiser):
    """
    SHAP beeswarm summary plot via ``shap.summary_plot()``.

    Compatible with all SHAP explainer algorithms.
    """

    def __init__(self, feature_names: list[str] | None = None, max_display: int = 20):
        self.feature_names = feature_names
        self.max_display = max_display

    def visualise(
        self, attributions: torch.Tensor, inputs: torch.Tensor | None = None, **kwargs: Any
    ) -> Figure:
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP visualiser is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

        values = _to_numpy(attributions)
        feats = _to_numpy(inputs) if inputs is not None else None

        fig = plt.figure()
        shap.summary_plot(
            values,
            features=feats,
            feature_names=self.feature_names,
            max_display=self.max_display,
            show=False,
            **kwargs,
        )
        return fig


class ShapWaterfallVisualiser(BaseVisualiser):
    """
    Per-sample SHAP waterfall chart via ``shap.plots.waterfall``.

    Compatible with all SHAP explainer algorithms.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        expected_value: float = 0.0,
        sample_index: int = 0,
        max_display: int = 10,
    ):
        """
        Args:
            feature_names:  Optional list of feature labels.
            expected_value: Model baseline / expected output value.
                            Most SHAP explainers expose this as
                            ``explainer.expected_value``.
            sample_index:   Which sample from the batch to visualise.
            max_display:    Maximum number of features to show.
        """
        self.feature_names = feature_names
        self.expected_value = expected_value
        self.sample_index = sample_index
        self.max_display = max_display

    def visualise(
        self, attributions: torch.Tensor, inputs: torch.Tensor | None = None, **kwargs: Any
    ) -> Figure:
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP visualiser is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

        values = _to_numpy(attributions)
        sample_vals = values[self.sample_index]

        feats = None
        if inputs is not None:
            feats = _to_numpy(inputs)[self.sample_index]

        feature_names = (
            self.feature_names if self.feature_names else [f"f{i}" for i in range(len(sample_vals))]
        )

        explanation = shap.Explanation(
            values=sample_vals,
            base_values=self.expected_value,
            data=feats,
            feature_names=feature_names,
        )

        shap.plots.waterfall(explanation, max_display=self.max_display, show=False, **kwargs)
        fig = plt.gcf()
        return _close_and_return(fig)


class ShapForceVisualiser(BaseVisualiser):
    """
    Per-sample SHAP force plot via ``shap.plots.force`` (matplotlib backend).

    Compatible with all SHAP explainer algorithms.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        expected_value: float = 0.0,
        sample_index: int = 0,
    ):
        self.feature_names = feature_names
        self.expected_value = expected_value
        self.sample_index = sample_index

    def visualise(
        self, attributions: torch.Tensor, inputs: torch.Tensor | None = None, **kwargs: Any
    ) -> Figure:
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP visualiser is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

        values = _to_numpy(attributions)
        sample_vals = values[self.sample_index]

        feats = None
        if inputs is not None:
            feats = _to_numpy(inputs)[self.sample_index]

        feature_names = (
            self.feature_names if self.feature_names else [f"f{i}" for i in range(len(sample_vals))]
        )

        shap.plots.force(
            base_value=self.expected_value,
            shap_values=sample_vals,
            features=feats,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            **kwargs,
        )
        fig = plt.gcf()
        return _close_and_return(fig)


# ---------------------------------------------------------------------------
# Image visualiser — RESTRICTED to GradientExplainer / DeepExplainer
# ---------------------------------------------------------------------------


class ShapImageVisualiser(BaseVisualiser):
    """
    Render image-level SHAP attributions with Matplotlib.

    This visualiser does not call ``shap.image_plot`` directly. Instead, it
    renders a RAITAP-managed figure that can optionally show the original
    image, a SHAP heatmap overlay, sample-aware titles, and a colorbar.

    .. warning::
       **Only compatible with ``GradientExplainer`` and ``DeepExplainer``.**
       These are the only SHAP explainers that compute pixel-level SHAP
       values suitable for image visualisation.
       Passing attributions from other explainers will produce meaningless
       plots.

    Positive contributions are shown in warm colours and negative
    contributions in cool colours, using the configured Matplotlib colormap.
    """

    compatible_algorithms: ClassVar[frozenset[str]] = frozenset(
        {"GradientExplainer", "DeepExplainer"}
    )

    def __init__(
        self,
        max_samples: int = 4,
        title: str | None = None,
        include_original_image: bool = True,
        show_colorbar: bool = True,
        cmap: str = "coolwarm",
        overlay_alpha: float = 0.65,
    ):
        """
        Args:
            max_samples: Maximum number of images to display side by side.
            title: Optional attribution panel title.
            include_original_image: Whether to render the original image next to the
                attribution heatmap when ``inputs`` are provided.
            show_colorbar: Whether to add a SHAP colorbar in the paired layout.
            cmap: Matplotlib colormap for the SHAP heatmap overlay.
            overlay_alpha: Alpha value used for the SHAP heatmap overlay.
        """
        self.max_samples = max_samples
        self.title = title
        self.include_original_image = include_original_image
        self.show_colorbar = show_colorbar
        self.cmap = cmap
        self.overlay_alpha = overlay_alpha

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        max_samples: int | None = None,
        sample_names: list[str] | None = None,
        show_sample_names: bool = False,
        title: str | None = None,
        algorithm: str | None = None,
        **kwargs: Any,
    ) -> Figure:
        """
        Args:
            attributions: ``(B, C, H, W)`` SHAP values tensor / array.
            inputs:        Original images ``(B, C, H, W)`` for background.
            max_samples:   Maximum number of images to display.
            sample_names:  Optional names per sample.
            show_sample_names: Whether to render sample names in subplot titles.
            title:         Optional attribution panel title. Overrides the algorithm-based
                           default title, even when set to an empty string.
            algorithm:     Optional explainer algorithm name used for the default
                           attribution title (rendered as ``"<Algorithm> (SHAP)"``).
            **kwargs:      Optional visual styling overrides.

        Returns:
            Matplotlib Figure.
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP visualiser is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

        del shap

        shap_vals = _to_numpy(attributions)  # (B, C, H, W)
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[np.newaxis]
        n = min(shap_vals.shape[0], self.max_samples if max_samples is None else max_samples)
        shap_vals = shap_vals[:n]

        pixel_values = None
        if inputs is not None:
            pv = _to_numpy(inputs)[:n]  # (B, C, H, W)
            pixel_values = np.transpose(pv, (0, 2, 3, 1))  # (B, H, W, C)
            pixel_values = np.stack([_normalise_image(image) for image in pixel_values], axis=0)

        names = [] if sample_names is None else [str(name) for name in sample_names[:n]]
        cmap = kwargs.pop("cmap", self.cmap)
        overlay_alpha = kwargs.pop("overlay_alpha", self.overlay_alpha)
        show_colorbar = bool(kwargs.pop("show_colorbar", self.show_colorbar))
        include_original_image = bool(
            kwargs.pop("include_original_image", self.include_original_image)
        )

        show_original_panel = include_original_image and pixel_values is not None
        if show_original_panel and show_colorbar:
            fig, axes = plt.subplots(
                n,
                3,
                figsize=(8.5, 4 * n),
                squeeze=False,
                gridspec_kw={"width_ratios": [1, 1, 0.05]},
            )
        elif show_original_panel:
            fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n), squeeze=False)
        elif show_colorbar:
            fig, axes = plt.subplots(
                n,
                2,
                figsize=(4.5, 4 * n),
                squeeze=False,
                gridspec_kw={"width_ratios": [1, 0.05]},
            )
        else:
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
            axes = np.array([axes], dtype=object) if n == 1 else np.asarray(axes, dtype=object)

        fallback_attr_title = (
            self.title if self.title is not None else _default_image_title(algorithm)
        )

        for i in range(n):
            shap_i = (
                np.transpose(shap_vals[i], (1, 2, 0)) if shap_vals[i].ndim == 3 else shap_vals[i]
            )
            heatmap = _image_heatmap(shap_i)
            max_abs = float(np.max(np.abs(heatmap))) if heatmap.size else 0.0
            if max_abs == 0.0:
                max_abs = 1.0

            sample_name = names[i] if show_sample_names and i < len(names) else None
            original_title = _compose_title("Original Image", sample_name)
            attr_title = _resolve_title(
                explicit_title=title,
                fallback_title=fallback_attr_title,
                sample_name=sample_name,
            )

            image_i = None if pixel_values is None else pixel_values[i]

            if show_original_panel:
                colorbar_ax = None
                if show_colorbar:
                    original_ax, attr_ax, colorbar_ax = axes[i]
                else:
                    original_ax, attr_ax = axes[i]

                if image_i is not None:
                    _display_image(original_ax, image_i)
                original_ax.set_title(original_title or "")
                original_ax.axis("off")
            else:
                if show_colorbar:
                    attr_ax, colorbar_ax = axes[i]
                else:
                    attr_ax = axes[i]
                    colorbar_ax = None

            if image_i is not None:
                _display_image(attr_ax, image_i)
            im = attr_ax.imshow(
                heatmap,
                cmap=cmap,
                alpha=overlay_alpha if image_i is not None else 1.0,
                vmin=-max_abs,
                vmax=max_abs,
            )
            attr_ax.set_title(attr_title or "")
            attr_ax.axis("off")

            if colorbar_ax is not None:
                fig.colorbar(im, cax=colorbar_ax)
                colorbar_ax.set_ylabel("SHAP value")

        return fig
