"""
RAITAP Transparency Module

Provides model explanation / attribution capabilities using SHAP and Captum.

Transparency Public Surface
---------------------------
Explainer classes expose `explainer.explain(model, inputs, **kwargs)`, which
returns an `ExplanationResult`. Each explanation can then render one
or more visualisations via `explanation.visualise(**kwargs)`.

Explainer classes (used as ``_target_`` values)
-----------------------------------------------
CaptumExplainer, ShapExplainer, AlibiExplainer (optional extra ``alibi``)

Visualiser classes (used as ``_target_`` values in visualisers list)
--------------------------------------------------------------------
CaptumImageVisualiser, CaptumTimeSeriesVisualiser, CaptumTextVisualiser
ShapBarVisualiser, ShapBeeswarmVisualiser, ShapWaterfallVisualiser,
ShapForceVisualiser, ShapImageVisualiser
TabularBarChartVisualiser
"""

from __future__ import annotations

from .contracts import ExplainerAdapter, ExplanationPayloadKind
from .exceptions import (
    ExplainerBackendIncompatibilityError,
    PayloadVisualiserIncompatibilityError,
    VisualiserIncompatibilityError,
)

# Explainer classes — public _target_ surface
from .explainers import AlibiExplainer, CaptumExplainer, CustomExplainer, ShapExplainer
from .factory import (
    Explanation,
    check_explainer_visualiser_compat,
    create_explainer,
    create_visualisers,
)

# Result objects
from .results import ConfiguredVisualiser, ExplanationResult, VisualisationResult

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
    "AlibiExplainer",
    "CaptumExplainer",
    "CustomExplainer",
    "ShapExplainer",
    # Result objects
    "ConfiguredVisualiser",
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
    # Contracts
    "ExplainerAdapter",
    "ExplanationPayloadKind",
    # Domain errors
    "ExplainerBackendIncompatibilityError",
    "PayloadVisualiserIncompatibilityError",
    "VisualiserIncompatibilityError",
    # Rest
    "Explanation",
    "create_explainer",
    "create_visualisers",
    "check_explainer_visualiser_compat",
]
