"""
RAITAP Transparency Module

Provides model explanation / attribution capabilities using SHAP and Captum.

Public API
----------
Explainer classes expose `explainer.explain(model, inputs, **kwargs)`, which
returns an `ExplanationResult`. Each explanation can then render one
visualisation at a time via `explanation.visualise(visualiser, **kwargs)`.

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
from .factory import (
    Explanation,
    check_explainer_visualiser_compat,
    create_explainer,
    create_visualisers,
)

# Result objects
from .results import ExplanationResult, VisualisationResult

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
    VisualiserIncompatibilityError,
)

__all__ = [  # noqa: RUF022
    # Explainer adapters
    "CaptumExplainer",
    "ShapExplainer",
    # Result objects
    "ExplanationResult",
    "VisualisationResult",
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
    # Rest
    "Explanation",
    "create_explainer",
    "create_visualisers",
    "check_explainer_visualiser_compat",
]
