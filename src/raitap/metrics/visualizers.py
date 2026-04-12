from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .base_metric import MetricResult

logger = logging.getLogger(__name__)


class MetricsVisualizer:
    """Generate matplotlib figures from MetricResult data."""

    @staticmethod
    def create_figures(result: MetricResult) -> dict[str, Figure]:
        """Generate charts for metrics.

        Returns dict with:
        - "metrics_overview": bar chart of all scalar metrics
        - "confusion_matrix": if confusion matrix exists in artifacts
        """
        figures = {}

        # Metrics overview bar chart
        if result.metrics:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            metric_names = list(result.metrics.keys())
            metric_values = [float(v) for v in result.metrics.values()]

            ax.bar(metric_names, metric_values)
            ax.set_xlabel("Metric")
            ax.set_ylabel("Value")
            ax.set_title("Metrics Overview")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()

            figures["metrics_overview"] = fig

        # Confusion matrix (if available)
        if "confusion_matrix" in result.artifacts:
            import numpy as np

            cm = result.artifacts["confusion_matrix"]
            if isinstance(cm, list | np.ndarray):
                fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
                im = ax.imshow(cm, cmap="Blues")
                ax.set_title("Confusion Matrix")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                figures["confusion_matrix"] = fig

        return figures
