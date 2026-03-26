"""SHAP-native visualisers (wrapping shap.plots.* and shap.summary_plot)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        plt.tight_layout()
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
        fig.tight_layout()
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
        fig.tight_layout()
        return _close_and_return(fig)


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
       plots.

    Red pixels increase the prediction; blue pixels decrease it.
    """

    compatible_algorithms: frozenset[str] = frozenset({"GradientExplainer", "DeepExplainer"})

    def __init__(self, max_samples: int = 4):
        """
        Args:
            max_samples: Maximum number of images to display side by side.
        """
        self.max_samples = max_samples

    def visualise(
        self, attributions: torch.Tensor, inputs: torch.Tensor | None = None, **kwargs: Any
    ) -> Figure:
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
            raise ImportError(
                "SHAP visualiser is enabled but shap is not installed. "
                "Install it with `uv sync --extra shap`."
            ) from e

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
