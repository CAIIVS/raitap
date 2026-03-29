"""Captum-native visualisers (wrapping captum.attr.visualization)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from .base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor or array-like to numpy."""
    val: Any = x
    if hasattr(val, "detach"):
        val = val.detach()
    if hasattr(val, "cpu"):
        val = val.cpu()
    if hasattr(val, "numpy"):
        val = val.numpy()
    return np.asarray(val)


class CaptumImageVisualiser(BaseVisualiser):
    """
    Visualise image attributions using ``captum.attr.visualization.visualize_image_attr``.

    Wraps the Captum native function so the output is a Matplotlib Figure
    that can be saved or returned by ``explain()``.

    Compatible with ALL Captum attribution algorithms.
    """

    def __init__(
        self,
        method: str = "blended_heat_map",
        sign: str = "all",
        show_colorbar: bool = True,
    ):
        """
        Args:
            method:       Captum viz method. Options:
                          ``"blended_heat_map"``, ``"heat_map"``,
                          ``"original_image"``, ``"masked_image"``,
                          ``"alpha_scaling"``.  Default: ``"blended_heat_map"``.
            sign:         Which signs to visualise.  Options:
                          ``"all"``, ``"positive"``, ``"negative"``,
                          ``"absolute_value"``.  Default: ``"all"``.
            show_colorbar: Whether to add a colorbar.  Default: ``True``.
        """
        self.method = method
        self.sign = sign
        self.show_colorbar = show_colorbar

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        max_samples: int = 8,
        sample_names: list[str] | None = None,
        show_sample_names: bool = False,
        **kwargs: Any,
    ) -> Figure:
        """
        Args:
            attributions: ``(B, C, H, W)`` or ``(B, H, W)`` tensor / array.
            inputs:       Original images ``(B, C, H, W)`` for overlay.
            max_samples:  Maximum number of samples to display (default: 8).
            sample_names: Optional names per sample (already normalised to stem IDs).
            show_sample_names: Whether to render sample names as subplot titles.
            **kwargs:     Forwarded to ``visualize_image_attr``.

        Returns:
            Matplotlib Figure with one column per sample.
        """
        try:
            from captum.attr import visualization as viz
        except ImportError as e:
            raise ImportError(
                "Captum visualiser is enabled but captum is not installed. "
                "Install it with `uv sync --extra captum`."
            ) from e

        attrs = _to_numpy(attributions)
        # Ensure batch dimension
        if attrs.ndim == 3:
            attrs = attrs[np.newaxis]  # (1, C, H, W)
        n = min(attrs.ndim and attrs.shape[0], max_samples)
        attrs = attrs[:n]

        origs = None
        if inputs is not None:
            origs = _to_numpy(inputs)
            if origs.ndim == 3:
                origs = origs[np.newaxis]
            origs = origs[:n]

        names = [] if sample_names is None else [str(name) for name in sample_names[:n]]

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        axes_list = [axes] if n == 1 else list(axes)

        for i, ax in enumerate(axes_list):
            # (C, H, W) -> (H, W, C)
            attr_i = np.transpose(attrs[i], (1, 2, 0)) if attrs[i].ndim == 3 else attrs[i]

            orig_i = None
            if origs is not None:
                orig_i = np.transpose(origs[i], (1, 2, 0)) if origs[i].ndim == 3 else origs[i]
                lo, hi = orig_i.min(), orig_i.max()
                if hi > lo:
                    orig_i = (orig_i - lo) / (hi - lo)

            # Let Captum render into our existing axes
            _, _ = viz.visualize_image_attr(
                attr_i,
                orig_i,
                method=self.method,
                sign=self.sign,
                show_colorbar=self.show_colorbar,
                plt_fig_axis=(fig, ax),
                use_pyplot=False,
                **kwargs,
            )
            if show_sample_names and i < len(names):
                base_title = ax.get_title().strip()
                label = f"{base_title}: {names[i]}" if base_title else names[i]
                ax.set_title(label, fontsize=9)

        fig.tight_layout()
        return fig


class CaptumTimeSeriesVisualiser(BaseVisualiser):
    """
    Visualise time-series attributions via
    ``captum.attr.visualization.visualize_timeseries_attr``.

    Compatible with ALL Captum attribution algorithms.
    """

    def __init__(
        self,
        method: str = "overlay_individual",
        sign: str = "absolute_value",
    ):
        """
        Args:
            method: One of ``"overlay_individual"``, ``"overlay_combined"``,
                    ``"colored_graph"``.  Default: ``"overlay_individual"``.
            sign:   One of ``"positive"``, ``"negative"``, ``"absolute_value"``,
                    ``"all"``.  Default: ``"absolute_value"``.
        """
        self.method = method
        self.sign = sign

    def visualise(
        self, attributions: torch.Tensor, inputs: torch.Tensor | None = None, **kwargs: Any
    ) -> Figure:
        """
        Args:
            attributions: ``(N, C)`` numpy array / tensor  (channels-last).
                          If a batch of shape ``(B, N, C)`` is given, the first
                          sample is used.
            inputs:       Matching ``(N, C)`` time-series data (optional).
            **kwargs:     Forwarded to ``visualize_timeseries_attr``.

        Returns:
            Matplotlib Figure.
        """
        try:
            from captum.attr import visualization as viz
        except ImportError as e:
            raise ImportError(
                "Captum visualiser is enabled but captum is not installed. "
                "Install it with `uv sync --extra captum`."
            ) from e

        attr = _to_numpy(attributions)
        # If batch dimension present, take first sample
        if attr.ndim == 3:
            attr = attr[0]

        data = None
        if inputs is not None:
            data = _to_numpy(inputs)
            if data.ndim == 3:
                data = data[0]

        if data is None:
            raise ValueError(
                "CaptumTimeSeriesVisualiser requires `inputs` (the original time-series "
                "data) to be passed alongside attributions."
            )

        fig, _ = viz.visualize_timeseries_attr(
            attr=attr,
            data=data,
            method=self.method,
            sign=self.sign,
            use_pyplot=False,
            **kwargs,
        )
        return fig


class CaptumTextVisualiser(BaseVisualiser):
    """
    Visualise per-token text attributions as a horizontal bar chart.

    This is a lightweight matplotlib-based implementation since Captum's native
    text visualisation renders HTML (not a Matplotlib Figure).

    Compatible with ALL Captum attribution algorithms on text/sequence inputs.

    Note: ``attributions`` should be a 1-D array of per-token scores for a
    single input. Pass ``token_labels`` via kwargs for readable output.
    """

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        token_labels: list[str] | None = None,
        **kwargs,
    ) -> Figure:
        """
        Args:
            attributions:  1-D attribution scores (one per token).
            inputs:        Ignored.
            token_labels:  List of token strings (optional).
            **kwargs:      Ignored (for API consistency).

        Returns:
            Matplotlib Figure with a horizontal bar chart of token importance.
        """
        attr = _to_numpy(attributions).ravel()
        n = len(attr)
        labels = token_labels if token_labels else [f"tok_{i}" for i in range(n)]

        fig, ax = plt.subplots(figsize=(8, max(4, n * 0.35)))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in attr]
        y = np.arange(n)
        ax.barh(y, attr, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Attribution")
        ax.set_title("Token Attribution")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig
