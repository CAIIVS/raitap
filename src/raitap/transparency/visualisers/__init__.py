"""Visualiser implementations for RAITAP transparency module."""

from __future__ import annotations

from .base import BaseVisualiser

# Captum-native visualisers
from .captum_visualisers import (
    CaptumImageVisualiser,
    CaptumTextVisualiser,
    CaptumTimeSeriesVisualiser,
)

# Legacy visualisers (kept for backward compatibility)
from .image_visualiser import ImageHeatmapvisualiser

# SHAP-native visualisers
from .shap_visualisers import (
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapForceVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
)
from .tabular_visualiser import TabularBarChartvisualiser

__all__ = [
    # Base
    "BaseVisualiser",
    # Legacy
    "ImageHeatmapvisualiser",
    "TabularBarChartvisualiser",
    # Captum
    "CaptumImageVisualiser",
    "CaptumTimeSeriesVisualiser",
    "CaptumTextVisualiser",
    # SHAP
    "ShapBarVisualiser",
    "ShapBeeswarmVisualiser",
    "ShapWaterfallVisualiser",
    "ShapForceVisualiser",
    "ShapImageVisualiser",
]
