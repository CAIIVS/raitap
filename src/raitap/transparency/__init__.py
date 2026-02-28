"""
RAITAP Transparency Module

Provides model explanation / attribution capabilities using SHAP and Captum.

Public API
----------
explain(config, model, inputs, **kwargs)
    One-call entry point.  Uses config to select the framework, algorithm,
    and visualisers; computes attributions; saves outputs; returns a dict
    with ``attributions`` and ``visualisations`` keys.

Lower-level helpers
-------------------
create_explainer(), method_from_config()
Captum, SHAP              – method registries
All Visualiser classes    – for direct use
"""

from __future__ import annotations

# Direct access for power users
from .explainers import CaptumExplainer, ShapExplainer

# Primary API
from .factory import create_explainer, explain, method_from_config
from .methods import SHAP, Captum, IntegratedGradients, KernelShap, Saliency  # Convenience aliases
from .visualisers import (
    CaptumImageVisualiser,
    CaptumTextVisualiser,
    CaptumTimeSeriesVisualiser,
    ImageHeatmapvisualiser,
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapForceVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
    TabularBarChartvisualiser,
)

__all__ = [
    # Primary API
    "explain",
    # Factory helpers
    "create_explainer",
    "method_from_config",
    # Registries
    "SHAP",
    "Captum",
    # Explainer adapters
    "CaptumExplainer",
    "ShapExplainer",
    # Captum visualisers
    "CaptumImageVisualiser",
    "CaptumTimeSeriesVisualiser",
    "CaptumTextVisualiser",
    # SHAP visualisers
    "ShapBarVisualiser",
    "ShapBeeswarmVisualiser",
    "ShapWaterfallVisualiser",
    "ShapForceVisualiser",
    "ShapImageVisualiser",
    # Legacy visualisers
    "ImageHeatmapvisualiser",
    "TabularBarChartvisualiser",
    # Aliases
    "IntegratedGradients",
    "KernelShap",
    "Saliency",
]
