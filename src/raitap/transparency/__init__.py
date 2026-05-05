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
CaptumExplainer, ShapExplainer

Visualiser classes (used as ``_target_`` values in visualisers list)
--------------------------------------------------------------------
CaptumImageVisualiser, CaptumTimeSeriesVisualiser, CaptumTextVisualiser
ShapBarVisualiser, ShapBeeswarmVisualiser, ShapWaterfallVisualiser,
ShapForceVisualiser, ShapImageVisualiser
TabularBarChartVisualiser
"""

from __future__ import annotations

from .contracts import (
    ExplainerAdapter,
    ExplainerCapability,
    ExplanationOutputSpace,
    ExplanationPayloadKind,
    ExplanationScope,
    ExplanationSemantics,
    ExplanationTarget,
    InputKind,
    InputSpec,
    MethodFamily,
    OutputSpaceSpec,
    SampleSelection,
    ScopeDefinitionStep,
    TensorLayout,
    VisualSummarySpec,
)
from .exceptions import (
    ExplainerBackendIncompatibilityError,
    PayloadVisualiserIncompatibilityError,
    VisualiserIncompatibilityError,
)
from .semantics import (
    explainer_capability,
    infer_input_spec,
    infer_output_space,
    method_families_for_explainer,
)


class _UnavailableOptionalDependency:
    def __init__(self, public_name: str, dependency: str) -> None:
        self._public_name = public_name
        self._dependency = dependency

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        self._raise()

    def __getattr__(self, _name: str) -> object:
        self._raise()

    def _raise(self) -> None:
        raise ImportError(f"{self._public_name} requires optional dependency {self._dependency!r}.")


def _unavailable(public_name: str, dependency: str) -> _UnavailableOptionalDependency:
    return _UnavailableOptionalDependency(public_name, dependency)


try:
    # Explainer classes — public _target_ surface
    from .explainers import (
        AbstractExplainer,
        AttributionOnlyExplainer,
        CaptumExplainer,
        FullExplainer,
        ShapExplainer,
    )
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
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    AbstractExplainer = _unavailable("AbstractExplainer", "torch")
    AttributionOnlyExplainer = _unavailable("AttributionOnlyExplainer", "torch")
    CaptumExplainer = _unavailable("CaptumExplainer", "torch")
    FullExplainer = _unavailable("FullExplainer", "torch")
    ShapExplainer = _unavailable("ShapExplainer", "torch")
    Explanation = _unavailable("Explanation", "torch")
    check_explainer_visualiser_compat = _unavailable("check_explainer_visualiser_compat", "torch")
    create_explainer = _unavailable("create_explainer", "torch")
    create_visualisers = _unavailable("create_visualisers", "torch")
    ConfiguredVisualiser = _unavailable("ConfiguredVisualiser", "torch")
    ExplanationResult = _unavailable("ExplanationResult", "torch")
    VisualisationResult = _unavailable("VisualisationResult", "torch")
    CaptumImageVisualiser = _unavailable("CaptumImageVisualiser", "torch")
    CaptumTextVisualiser = _unavailable("CaptumTextVisualiser", "torch")
    CaptumTimeSeriesVisualiser = _unavailable("CaptumTimeSeriesVisualiser", "torch")
    ShapBarVisualiser = _unavailable("ShapBarVisualiser", "torch")
    ShapBeeswarmVisualiser = _unavailable("ShapBeeswarmVisualiser", "torch")
    ShapForceVisualiser = _unavailable("ShapForceVisualiser", "torch")
    ShapImageVisualiser = _unavailable("ShapImageVisualiser", "torch")
    ShapWaterfallVisualiser = _unavailable("ShapWaterfallVisualiser", "torch")
    TabularBarChartVisualiser = _unavailable("TabularBarChartVisualiser", "torch")

__all__ = [  # noqa: RUF022
    # Explainer adapters
    "CaptumExplainer",
    "AbstractExplainer",
    "AttributionOnlyExplainer",
    "FullExplainer",
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
    "ExplainerCapability",
    "ExplanationOutputSpace",
    "ExplanationPayloadKind",
    "ExplanationScope",
    "ExplanationSemantics",
    "ExplanationTarget",
    "InputKind",
    "InputSpec",
    "MethodFamily",
    "OutputSpaceSpec",
    "SampleSelection",
    "ScopeDefinitionStep",
    "TensorLayout",
    "VisualSummarySpec",
    # Semantic helpers
    "explainer_capability",
    "infer_input_spec",
    "infer_output_space",
    "method_families_for_explainer",
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
