"""Tabular modality visualization (feature importance bars)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from .base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure


class TabularBarChartVisualiser(BaseVisualiser):
    """
    Visualise attributions for tabular data as bar charts.

    Works with any attribution method (Captum, SHAP, etc.)
    """

    report_scope: ClassVar[str] = "global"

    def __init__(self, feature_names: list[str] | None = None):
        """
        Args:
            feature_names: List of feature names for x-axis labels
        """
        self.feature_names = feature_names

    def visualise(
        self, attributions: torch.Tensor, inputs: torch.Tensor | None = None, **kwargs: Any
    ) -> Figure:
        """
        Create feature importance bar chart.

        Args:
            attributions: (B, num_features) array
            inputs: Not used for tabular visualization

        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        if hasattr(attributions, "detach"):
            attrs_np = attributions.detach().cpu().numpy()
        elif hasattr(attributions, "numpy"):
            attrs_np = attributions.cpu().numpy()
        else:
            attrs_np = np.array(attributions)

        # Aggregate across batch (mean absolute attribution)
        mean_importance = np.abs(attrs_np).mean(axis=0)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(mean_importance))
        ax.bar(x, mean_importance)

        if self.feature_names:
            ax.set_xticks(x)
            ax.set_xticklabels(self.feature_names, rotation=45, ha="right")

        ax.set_ylabel("Mean Absolute Attribution")
        ax.set_xlabel("Features")
        ax.set_title("Feature Importance")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        return fig
