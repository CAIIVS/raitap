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

Module layout (for contributors):

- ``phase.py`` — pipeline entry point: ``TransparencyPhase`` (what the registry
  assembles) + the ``assess_transparency`` work fn. Start here to follow a run.
- ``explain_detection.py`` — detection-task per-box K-loop (one result per box).
- ``factory.py`` — builds explainer + visualiser instances from config.
- ``results.py`` — ``ExplanationResult`` (owns its ``.visualisations``) + ``VisualisationResult``.
- ``report.py`` — ``TransparencyPhaseResult`` + report-section builders.
- ``explainers/`` — the XAI adapters (Captum, SHAP).
- ``visualisers/`` — the figure renderers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.utils.errors import BackendIncompatibilityError

from .contracts import (
    ExplainerAdapter,
    ExplainerCapability,
    ExplainerSemanticsHints,
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
        DetectionImageVisualiser,
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
    check_explainer_visualiser_compat = _unavailable("check_explainer_visualiser_compat", "torch")
    create_explainer = _unavailable("create_explainer", "torch")
    create_visualisers = _unavailable("create_visualisers", "torch")
    ConfiguredVisualiser = _unavailable("ConfiguredVisualiser", "torch")
    ExplanationResult = _unavailable("ExplanationResult", "torch")
    VisualisationResult = _unavailable("VisualisationResult", "torch")
    CaptumImageVisualiser = _unavailable("CaptumImageVisualiser", "torch")
    CaptumTextVisualiser = _unavailable("CaptumTextVisualiser", "torch")
    CaptumTimeSeriesVisualiser = _unavailable("CaptumTimeSeriesVisualiser", "torch")
    DetectionImageVisualiser = _unavailable("DetectionImageVisualiser", "torch")
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

    # Real submodules (report, phase, factory, results, ...) resolve lazily as
    # package attributes so dotted-path access / monkeypatch agree with
    # ``from ... import`` resolution. Imported on access (not eagerly) to avoid a
    # load-time cycle via reporting -> robustness.contracts.
    import importlib

    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError:
        pass

    from raitap._adapters import lookup

    try:
        return lookup("transparency", name)
    except AttributeError:
        from raitap.configs import register_configs

        register_configs()  # idempotent; fires in-tree imports + plugin discovery
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
    # Detection visualisers
    "DetectionImageVisualiser",
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
    "ExplainerSemanticsHints",
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
    "BackendIncompatibilityError",
    "PayloadVisualiserIncompatibilityError",
    "VisualiserIncompatibilityError",
    # Rest
    "create_explainer",
    "create_visualisers",
    "check_explainer_visualiser_compat",
]
