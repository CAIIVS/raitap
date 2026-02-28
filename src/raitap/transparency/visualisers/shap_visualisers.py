"""SHAP-native visualisers (wrapping shap.plots.* and shap.summary_plot)"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .base import BaseVisualiser


def _to_numpy(x) -> np.ndarray:
    """Convert tensor or array-like to numpy (float32)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32)


def _close_and_return(fig: Figure) -> Figure:
    """Detach a figure from pyplot's interactive state then return it."""
    plt.close(fig)
    return fig


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

    def visualise(self, attributions, inputs=None, **kwargs) -> Figure:
        """
        Args:
            attributions: ``(B, F)`` SHAP values tensor / array.
            inputs:       Original feature values ``(B, F)`` (used for colouring).
            **kwargs:     Forwarded to ``shap.summary_plot``.
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError("SHAP not installed.  pip install shap>=0.46.0") from e

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
        plt.tight_layout()
        return fig


class ShapBeeswarmVisualiser(BaseVisualiser):
    """
    SHAP beeswarm summary plot via ``shap.summary_plot()``.

    Compatible with all SHAP explainer algorithms.
    """

    def __init__(self, feature_names: list[str] | None = None, max_display: int = 20):
        self.feature_names = feature_names
        self.max_display = max_display

    def visualise(self, attributions, inputs=None, **kwargs) -> Figure:
        try:
            import shap
        except ImportError as e:
            raise ImportError("SHAP not installed.  pip install shap>=0.46.0") from e

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
        plt.tight_layout()
        return fig


class ShapWaterfallVisualiser(BaseVisualiser):
    """
    Per-sample SHAP waterfall chart (first sample in batch).

    Renders a matplotlib-based waterfall chart showing how each feature
    contribution moves the output from the base value to the final prediction.
    Compatible with all SHAP explainer algorithms.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        expected_value: float = 0.0,
        sample_index: int = 0,
    ):
        """
        Args:
            feature_names:  Optional list of feature labels.
            expected_value: Model baseline / expected output value.
                            Most SHAP explainers expose this as
                            ``explainer.expected_value``.
            sample_index:   Which sample from the batch to visualise.
        """
        self.feature_names = feature_names
        self.expected_value = expected_value
        self.sample_index = sample_index

    def visualise(self, attributions, inputs=None, **kwargs) -> Figure:
        values = _to_numpy(attributions)
        sample_vals = values[self.sample_index]

        # Build a simple matplotlib waterfall chart.
        n_features = len(sample_vals)
        labels = self.feature_names if self.feature_names else [f"f{i}" for i in range(n_features)]
        # Sort by magnitude
        order = np.argsort(np.abs(sample_vals))[::-1]
        sorted_vals = sample_vals[order]
        sorted_labels = [labels[i] for i in order]

        fig, ax = plt.subplots(figsize=(8, max(4, n_features * 0.3)))
        running = self.expected_value
        lefts, heights, colors = [], [], []
        for val in sorted_vals:
            lefts.append(running)
            heights.append(val)
            colors.append("#e74c3c" if val > 0 else "#3498db")
            running += val
        y = np.arange(n_features)
        ax.barh(y, heights, left=lefts, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_labels, fontsize=8)
        ax.axvline(self.expected_value, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Output value")
        ax.set_title(
            f"Waterfall (sample {self.sample_index}) "
            f"- base: {self.expected_value:.4f}, "
            f"final: {running:.4f}"
        )
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig


class ShapForceVisualiser(BaseVisualiser):
    """
    Per-sample SHAP force plot (first sample in batch) via ``shap.force_plot``.

    Compatible with all SHAP explainer algorithms.

    .. note::
       ``shap.force_plot`` renders as an inline HTML object in notebooks.
       This visualiser converts it to a static Matplotlib figure using
       ``matplotlib`` text rendering as fallback.
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

    def visualise(self, attributions, inputs=None, **kwargs) -> Figure:
        values = _to_numpy(attributions)
        sample_vals = values[self.sample_index]

        # shap.force_plot returns an AdditiveForceVisualizer (HTML).
        # We render the attribution as a bar chart instead so the result
        # is a standard Matplotlib figure (saves cleanly to PNG/PDF).
        n_features = len(sample_vals)
        x = np.arange(n_features)
        labels = self.feature_names if self.feature_names else [f"f{i}" for i in range(n_features)]

        fig, ax = plt.subplots(figsize=(max(8, n_features * 0.5), 4))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in sample_vals]
        ax.bar(x, sample_vals, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("SHAP value")
        ax.set_title(
            f"Force plot (sample {self.sample_index}) - base value: {self.expected_value:.4f}"
        )
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Image visualiser — RESTRICTED to GradientExplainer / DeepExplainer
# ---------------------------------------------------------------------------


class ShapImageVisualiser(BaseVisualiser):
    """
    Image-level SHAP attribution plot via ``shap.image_plot``.

    .. warning::
       **Only compatible with ``GradientExplainer`` and ``DeepExplainer``.**
       These are the only SHAP explainers that compute pixel-level SHAP
       values suitable for image visualisation.
       Passing attributions from other explainers will produce meaningless
       plots.  Use the registry to enforce compatibility at config time.

    Red pixels increase the prediction; blue pixels decrease it.
    """

    def __init__(self, max_samples: int = 4):
        """
        Args:
            max_samples: Maximum number of images to display side by side.
        """
        self.max_samples = max_samples

    def visualise(self, attributions, inputs=None, **kwargs) -> Figure:
        """
        Args:
            attributions: ``(B, C, H, W)`` SHAP values tensor / array.
            inputs:        Original images ``(B, C, H, W)`` for background.
            **kwargs:      Forwarded to ``shap.image_plot``.

        Returns:
            Matplotlib Figure.
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError("SHAP not installed.  pip install shap>=0.46.0") from e

        shap_vals = _to_numpy(attributions)  # (B, C, H, W)
        n = min(shap_vals.shape[0], self.max_samples)
        shap_vals = shap_vals[:n]

        # shap.image_plot expects (B, H, W, C)
        shap_vals_hwc = np.transpose(shap_vals, (0, 2, 3, 1))

        pixel_values = None
        if inputs is not None:
            pv = _to_numpy(inputs)[:n]  # (B, C, H, W)
            pixel_values = np.transpose(pv, (0, 2, 3, 1))  # (B, H, W, C)
            # Normalise to [0, 1]
            lo, hi = pixel_values.min(), pixel_values.max()
            if hi > lo:
                pixel_values = (pixel_values - lo) / (hi - lo)

        shap.image_plot(
            shap_vals_hwc,
            pixel_values,
            show=False,
            **kwargs,
        )
        fig = plt.gcf()
        plt.tight_layout()
        return fig
