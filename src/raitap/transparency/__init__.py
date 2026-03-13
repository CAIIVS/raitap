"""
RAITAP Transparency Module

Provides model explanation / attribution capabilities using SHAP and Captum.

Public API
----------
explain(config, model, inputs, **kwargs)
    One-call entry point.  Reads ``_target_`` / ``algorithm`` / ``visualisers``
    from the config, instantiates the appropriate explainer and visualisers via
    Hydra's ``instantiate()``, and returns a dict with ``attributions``,
    ``visualisations``, and ``run_dir``.

Explainer classes (used as ``_target_`` values)
-----------------------------------------------
CaptumExplainer, ShapExplainer

Visualiser classes (used as ``_target_`` values in visualisers list)
--------------------------------------------------------------------
CaptumImageVisualiser, CaptumTimeSeriesVisualiser, CaptumTextVisualiser
ShapBarVisualiser, ShapBeeswarmVisualiser, ShapWaterfallVisualiser,
ShapForceVisualiser, ShapImageVisualiser
TabularBarChartVisualiser
"""

from __future__ import annotations

# Explainer classes — public _target_ surface
from .explainers import CaptumExplainer, ShapExplainer

# Primary API
from .factory import explain, explain_and_log

# Domain error type
from .methods_registry import VisualiserIncompatibilityError

# Visualiser classes — public _target_ surface
from .visualisers import (
    CaptumImageVisualiser,
    CaptumTextVisualiser,
    CaptumTimeSeriesVisualiser,
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapForceVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
    TabularBarChartVisualiser,
)

__all__ = [  # noqa: RUF022
    # Explainer adapters
    "CaptumExplainer",
    "ShapExplainer",
    # Captum visualisers
    "CaptumImageVisualiser",
    "CaptumTextVisualiser",
    "CaptumTimeSeriesVisualiser",
    # SHAP visualisers
    "ShapBarVisualiser",
    "ShapBeeswarmVisualiser",
    "ShapForceVisualiser",
    "ShapImageVisualiser",
    "ShapWaterfallVisualiser",
    # Framework-agnostic
    "TabularBarChartVisualiser",
    # Domain errors
    "VisualiserIncompatibilityError",
    # Primary API
    "explain",
    "explain_and_log",
]
