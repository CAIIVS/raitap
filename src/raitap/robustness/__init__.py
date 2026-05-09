"""
RAITAP Robustness Module

Provides per-sample robustness assessments under a perturbation budget. The
module distinguishes two complementary methods:

* **Empirical attacks** — try to find an adversarial example (torchattacks, foolbox).
* **Formal verification** — prove no adversarial example exists (auto_LiRPA,
  alpha-beta-CROWN; arrives in a follow-up PR — the module shape already accommodates it).

Public Surface
--------------
Assessor classes (``_target_`` values; live under ``raitap.robustness.assessors.``):
    TorchattacksAssessor, FoolboxAssessor

Visualiser classes (``_target_`` values; live under ``raitap.robustness.visualisers.``):
    ImagePairVisualiser, PerturbationHeatmapVisualiser
"""

from __future__ import annotations

from .contracts import (
    VERDICT_CODES,
    VERDICT_FROM_CODE,
    AssessorAdapter,
    MethodKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessSemantics,
    RobustnessVerdict,
    RobustnessVisualisationContext,
    ThreatModel,
    VerificationOutcome,
    decode_verdict,
    encode_verdict,
)
from .exceptions import (
    AssessorBackendIncompatibilityError,
    MethodKindVisualiserIncompatibilityError,
    MissingTargetsError,
    RobustnessVisualiserIncompatibilityError,
)
from .semantics import (
    AssessorSemanticsHints,
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


__all__ = [  # noqa: RUF022
    # Assessor classes
    "BaseAssessor",
    "EmpiricalAttackAssessor",
    "FormalVerificationAssessor",
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
    "MethodKind",
    "Objective",
    "PerturbationBudget",
    "PerturbationNorm",
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
    "AssessorBackendIncompatibilityError",
    "MethodKindVisualiserIncompatibilityError",
    "MissingTargetsError",
    "RobustnessVisualiserIncompatibilityError",
    # Semantics
    "AssessorSemanticsHints",
    "assessor_semantics",
    "hints_for_assessor",
    # Factory
    "RobustnessAssessment",
    "check_assessor_visualiser_compat",
    "create_assessor",
    "create_robustness_visualisers",
]
