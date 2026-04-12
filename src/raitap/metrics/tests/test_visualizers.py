"""Tests for metrics visualizers."""

from __future__ import annotations

from raitap.metrics.base_metric import MetricResult
from raitap.metrics.visualizers import MetricsVisualizer


def test_metrics_visualizer_creates_overview_chart() -> None:
    """Test basic chart generation from metrics."""
    result = MetricResult(
        metrics={"accuracy": 0.95, "precision": 0.92, "recall": 0.88},
        artifacts={},
    )

    figures = MetricsVisualizer.create_figures(result)

    assert "metrics_overview" in figures
    assert figures["metrics_overview"] is not None
    # Verify it's a matplotlib Figure
    assert hasattr(figures["metrics_overview"], "savefig")


def test_metrics_visualizer_empty_metrics() -> None:
    """Test that empty metrics doesn't crash."""
    result = MetricResult(
        metrics={},
        artifacts={},
    )

    figures = MetricsVisualizer.create_figures(result)

    assert isinstance(figures, dict)
    # Should return empty dict when no metrics
    assert len(figures) == 0


def test_metrics_visualizer_with_confusion_matrix() -> None:
    """Test confusion matrix visualization."""
    import numpy as np

    cm = np.array([[10, 2], [3, 15]])
    result = MetricResult(
        metrics={"accuracy": 0.89},
        artifacts={"confusion_matrix": cm},
    )

    figures = MetricsVisualizer.create_figures(result)

    assert "metrics_overview" in figures
    assert "confusion_matrix" in figures
    assert figures["confusion_matrix"] is not None
