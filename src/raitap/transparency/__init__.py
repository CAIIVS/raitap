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

from typing import TYPE_CHECKING, Any

from .contracts import (
    ALL,
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
        AttributionOnlyExplainer,
        BaseExplainer,
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
    BaseExplainer = _unavailable("BaseExplainer", "torch")
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


if TYPE_CHECKING:
    from raitap.configs.schema import TransparencyConfig


def __getattr__(name: str) -> Any:
    """Resolve hydra-zen builders by their ``registry_name``, plus the
    schema dataclass (:class:`~raitap.configs.schema.TransparencyConfig`)
    re-exported here so the module is the single owner of both the type
    and its builder instances.

    Lets users write::

        from raitap.transparency import TransparencyConfig, captum, captum_image

    without us hand-maintaining a registry — the builder is created by
    :class:`raitap._adapters.AdapterMixin` at class-declaration time and
    looked up here.
    """
    if name == "TransparencyConfig":
        from raitap.configs.schema import TransparencyConfig

        return TransparencyConfig
    from raitap._adapters import lookup

    return lookup("transparency", name)


__all__ = [  # noqa: RUF022
    # Schema dataclass (lazy)
    "TransparencyConfig",
    # Explainer adapters
    "CaptumExplainer",
    "BaseExplainer",
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
