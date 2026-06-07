"""
RAITAP Robustness Module

Provides per-sample robustness assessments under a perturbation budget. The
module distinguishes two complementary methods:

* **Empirical attacks** — try to find an adversarial example (torchattacks, foolbox).
* **Formal verification** — prove no adversarial example exists (Marabou complete
  SMT; auto_LiRPA sound+incomplete bound propagation).

Public Surface
--------------
Assessor classes (``_target_`` values; live under ``raitap.robustness.assessors.``):
    TorchattacksAssessor, FoolboxAssessor

Visualiser classes (``_target_`` values; live under ``raitap.robustness.visualisers.``):
    ImagePairVisualiser, PerturbationHeatmapVisualiser

Module layout (for contributors):

- ``phase.py`` — pipeline entry point: ``RobustnessPhase`` (what the registry
  assembles) + the ``assess_robustness`` work fn + target resolution. Start here.
- ``factory.py`` — builds assessor instances from config.
- ``results.py`` — ``RobustnessResult`` (owns its ``.visualisations``) +
  ``RobustnessVisualisationResult``.
- ``report.py`` — ``RobustnessPhaseResult`` + report-section builders.
- ``assessors/`` — the attack / verification adapters.
- ``visualisers/`` — the figure renderers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.utils.errors import BackendIncompatibilityError

from .contracts import (
    VERDICT_CODES,
    VERDICT_FROM_CODE,
    AssessmentKind,
    AssessorAdapter,
    Objective,
    PerturbationBudget,
    PerturbationDistribution,
    PerturbationNorm,
    PerturbationRegion,
    ReportFigureScope,
    RobustnessCase,
    RobustnessSemantics,
    RobustnessVerdict,
    RobustnessVisualisationContext,
    ThreatModel,
    VerificationOutcome,
    decode_verdict,
    encode_verdict,
)
from .exceptions import (
    AssessmentKindVisualiserIncompatibilityError,
    AssessorBackendIncompatibilityError,
    MissingTargetsError,
    RobustnessVisualiserIncompatibilityError,
)
from .semantics import (
    AssessorAlgorithmSpec,
    assessor_semantics,
    hints_for_assessor,
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
    from .assessors import (
        BaseAssessor,
        EmpiricalAttackAssessor,
        FoolboxAssessor,
        FormalVerificationAssessor,
        ImageCorruptionsAssessor,
        TorchattacksAssessor,
    )
    from .factory import (
        RobustnessAssessment,
        check_assessor_visualiser_compat,
        create_assessor,
        create_robustness_visualisers,
    )
    from .results import (
        ConfiguredRobustnessVisualiser,
        RobustnessMetrics,
        RobustnessResult,
        RobustnessVisualisationResult,
        decode_verdicts,
        encode_verdicts,
    )
    from .visualisers import (
        BaseRobustnessVisualiser,
        ImagePairVisualiser,
        PerturbationHeatmapVisualiser,
    )
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    BaseAssessor = _unavailable("BaseAssessor", "torch")
    EmpiricalAttackAssessor = _unavailable("EmpiricalAttackAssessor", "torch")
    FormalVerificationAssessor = _unavailable("FormalVerificationAssessor", "torch")
    TorchattacksAssessor = _unavailable("TorchattacksAssessor", "torch")
    FoolboxAssessor = _unavailable("FoolboxAssessor", "torch")
    ImageCorruptionsAssessor = _unavailable("ImageCorruptionsAssessor", "torch")
    RobustnessAssessment = _unavailable("RobustnessAssessment", "torch")
    check_assessor_visualiser_compat = _unavailable("check_assessor_visualiser_compat", "torch")
    create_assessor = _unavailable("create_assessor", "torch")
    create_robustness_visualisers = _unavailable("create_robustness_visualisers", "torch")
    ConfiguredRobustnessVisualiser = _unavailable("ConfiguredRobustnessVisualiser", "torch")
    RobustnessMetrics = _unavailable("RobustnessMetrics", "torch")
    RobustnessResult = _unavailable("RobustnessResult", "torch")
    RobustnessVisualisationResult = _unavailable("RobustnessVisualisationResult", "torch")
    decode_verdicts = _unavailable("decode_verdicts", "torch")
    encode_verdicts = _unavailable("encode_verdicts", "torch")
    BaseRobustnessVisualiser = _unavailable("BaseRobustnessVisualiser", "torch")
    ImagePairVisualiser = _unavailable("ImagePairVisualiser", "torch")
    PerturbationHeatmapVisualiser = _unavailable("PerturbationHeatmapVisualiser", "torch")


if TYPE_CHECKING:
    from raitap.configs.schema import RobustnessConfig


def __getattr__(name: str) -> Any:
    """Resolve hydra-zen builders (assessors + visualisers) by registry name,
    plus the schema dataclass (:class:`~raitap.configs.schema.RobustnessConfig`)
    re-exported here so the module owns both the type and its instances."""
    if name == "RobustnessConfig":
        from raitap.configs.schema import RobustnessConfig

        return RobustnessConfig
    from raitap._adapters import lookup

    try:
        return lookup("robustness", name)
    except AttributeError:
        from raitap.configs import register_configs

        register_configs()  # idempotent; fires in-tree imports + plugin discovery
        return lookup("robustness", name)


__all__ = [  # noqa: RUF022
    # Schema dataclass (lazy)
    "RobustnessConfig",
    # Assessor classes
    "BaseAssessor",
    "EmpiricalAttackAssessor",
    "FormalVerificationAssessor",
    "ImageCorruptionsAssessor",
    "TorchattacksAssessor",
    "FoolboxAssessor",
    # Visualisers
    "BaseRobustnessVisualiser",
    "ImagePairVisualiser",
    "PerturbationHeatmapVisualiser",
    # Result objects
    "ConfiguredRobustnessVisualiser",
    "RobustnessMetrics",
    "RobustnessResult",
    "RobustnessVisualisationResult",
    "decode_verdicts",
    "encode_verdicts",
    # Contracts
    "AssessorAdapter",
    "AssessmentKind",
    "Objective",
    "PerturbationBudget",
    "PerturbationDistribution",
    "PerturbationNorm",
    "PerturbationRegion",
    "ReportFigureScope",
    "RobustnessCase",
    "RobustnessSemantics",
    "RobustnessVerdict",
    "RobustnessVisualisationContext",
    "ThreatModel",
    "VerificationOutcome",
    "VERDICT_CODES",
    "VERDICT_FROM_CODE",
    "decode_verdict",
    "encode_verdict",
    # Exceptions
    "BackendIncompatibilityError",
    "AssessorBackendIncompatibilityError",
    "AssessmentKindVisualiserIncompatibilityError",
    "MissingTargetsError",
    "RobustnessVisualiserIncompatibilityError",
    # Semantics
    "AssessorAlgorithmSpec",
    "assessor_semantics",
    "hints_for_assessor",
    # Factory
    "RobustnessAssessment",
    "check_assessor_visualiser_compat",
    "create_assessor",
    "create_robustness_visualisers",
]
