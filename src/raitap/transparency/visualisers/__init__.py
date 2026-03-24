"""Visualiser implementations for RAITAP transparency module."""

from __future__ import annotations

from .base_visualiser import BaseVisualiser

# Captum-native visualisers
from .captum_visualisers import (
    CaptumImageVisualiser,
    CaptumTextVisualiser,
    CaptumTimeSeriesVisualiser,
)

# SHAP-native visualisers
from .shap_visualisers import (
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapForceVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
)
from .tabular_visualiser import TabularBarChartVisualiser

__all__ = [
    # Base
    "BaseVisualiser",
    # Captum
    "CaptumImageVisualiser",
    "CaptumTextVisualiser",
    "CaptumTimeSeriesVisualiser",
    # SHAP
    "ShapBarVisualiser",
    "ShapBeeswarmVisualiser",
    "ShapForceVisualiser",
    "ShapImageVisualiser",
    "ShapWaterfallVisualiser",
    # Framework-agnostic
    "TabularBarChartVisualiser",
]
