"""Visualiser implementations for RAITAP transparency module."""

from __future__ import annotations

from raitap.transparency.exceptions import VisualiserIncompatibilityError

from .base_visualiser import BaseVisualiser

# Captum-native visualisers
from .captum_visualisers import (
    CaptumImageVisualiser,
    CaptumTextVisualiser,
    CaptumTimeSeriesVisualiser,
)

# Detection visualisers
from .detection_image_visualiser import DetectionImageVisualiser
from .input_thumbnail import InputThumbnailVisualiser

# SHAP-native visualisers
from .shap_visualisers import (
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapForceVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
)
from .tabular_visualiser import TabularBarChartVisualiser

__all__ = [  # noqa: RUF022
    # Base
    "BaseVisualiser",
    "VisualiserIncompatibilityError",
    "InputThumbnailVisualiser",
    # Captum
    "CaptumImageVisualiser",
    "CaptumTextVisualiser",
    "CaptumTimeSeriesVisualiser",
    # Detection
    "DetectionImageVisualiser",
    # SHAP
    "ShapBarVisualiser",
    "ShapBeeswarmVisualiser",
    "ShapForceVisualiser",
    "ShapImageVisualiser",
    "ShapWaterfallVisualiser",
    # Framework-agnostic
    "TabularBarChartVisualiser",
]
