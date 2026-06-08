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

# Layer-activation visualiser
from .layer_activation_visualiser import LayerActivationVisualiser

# SHAP-native visualisers
from .shap_visualisers import (
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapForceVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
)
from .structured_payload_visualiser import StructuredPayloadSummaryVisualiser
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
    "LayerActivationVisualiser",
    "StructuredPayloadSummaryVisualiser",
    "TabularBarChartVisualiser",
]
