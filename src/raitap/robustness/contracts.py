"""
Shared robustness contracts: assessment-kind taxonomy, threat model, and verdict typing.

The robustness module distinguishes three fundamentally different ways to assess a
model's robustness against perturbations:

* ``AssessmentKind.EMPIRICAL_ATTACK`` — try to find an adversarial example within a
  perturbation budget (torchattacks, foolbox, …). A "non-attack" outcome does
  *not* prove robustness; it just means the configured attack failed.
* ``AssessmentKind.FORMAL_VERIFICATION`` — prove (or refute) that no adversarial
  example exists within the budget (auto_LiRPA, alpha-beta-CROWN, ...). Outcomes are
  verified / falsified / unknown.
* ``AssessmentKind.STATISTICAL_SAMPLING`` — measure average-case accuracy under a
  perturbation distribution (ImageNet-C corruptions, …). Outcomes are per-sample
  correct / misclassified.

Each procedure belongs to exactly one ``RobustnessCase`` (worst-case or average-case),
derived via ``case_for(kind)``.  A single ``RobustnessSemantics`` carries this
distinction so that downstream result handling, reporting, and visualiser-compatibility
checks can branch on ``assessment_kind`` instead of duck-typing.
"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from collections.abc import Set as AbstractSet  # noqa: TC003
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from raitap.transparency.contracts import InputSpec, SampleSelection  # noqa: TC001

if TYPE_CHECKING:
    from raitap.reproducibility import Seeding

ConfiguredRobustnessVisualiser = Any
RobustnessResult = Any
Module = Any
Tensor = Any


class AssessmentKind(StrEnum):
    """LEVEL 1 — procedure-level taxonomy; coarsens into RobustnessCase via case_for().

    Distinguishes three assessment procedures:
    * EMPIRICAL_ATTACK — adversarial example search (worst-case).
    * FORMAL_VERIFICATION — sound proof or refutation (worst-case).
    * STATISTICAL_SAMPLING — accuracy under a perturbation distribution (average-case).
    """

    EMPIRICAL_ATTACK = "empirical_attack"
    FORMAL_VERIFICATION = "formal_verification"
    STATISTICAL_SAMPLING = "statistical_sampling"


class RobustnessCase(StrEnum):
    """LEVEL 2 — the worst/average 'case' from thesis §2.4; a coarsening of AssessmentKind.

    Not an independent axis: every procedure belongs to exactly one case and no
    procedure spans cases, so case is *derived* from kind, never stored.
    """

    WORST_CASE = "worst_case"
    AVERAGE_CASE = "average_case"


_CASE_BY_KIND: dict[AssessmentKind, RobustnessCase] = {
    AssessmentKind.EMPIRICAL_ATTACK: RobustnessCase.WORST_CASE,
    AssessmentKind.FORMAL_VERIFICATION: RobustnessCase.WORST_CASE,
    AssessmentKind.STATISTICAL_SAMPLING: RobustnessCase.AVERAGE_CASE,
}


def case_for(kind: AssessmentKind) -> RobustnessCase:
    """Return the robustness case a procedure belongs to."""
    return _CASE_BY_KIND[kind]


class ReportFigureScope(StrEnum):
    """Where a robustness visualiser's figure belongs in the report layout.

    Read by the reporting layer to place each staged figure: ``ASSESSOR`` figures
    summarise the whole assessment (one chart per assessor — e.g. clean-vs-corrupted
    accuracy, verdict-count summary, output-bound cohorts); ``PER_SAMPLE`` figures
    show one input each (e.g. original/perturbed image pairs). Defaults to
    ``PER_SAMPLE`` on the visualiser base so empirical image visualisers keep their
    existing per-sample placement without opting in.
    """

    ASSESSOR = "assessor"
    PER_SAMPLE = "per_sample"


class RobustnessVerdict(StrEnum):
    """Per-sample assessment outcome.

    Empirical assessors emit ``ATTACK_SUCCEEDED`` / ``ATTACK_FAILED`` (the latter
    does NOT prove robustness — it only means the configured attack failed to find
    an adversarial example within the budget).
    Formal verification assessors emit ``VERIFIED`` / ``FALSIFIED`` / ``UNKNOWN``
    (and ``ERROR`` for per-sample crashes / timeouts).
    Statistical-sampling assessors emit ``CORRECT_UNDER_PERTURBATION`` /
    ``MISCLASSIFIED_UNDER_PERTURBATION`` (whether the corrupted input was still
    classified as its ground-truth label).
    """

    ATTACK_SUCCEEDED = "attack_succeeded"
    ATTACK_FAILED = "attack_failed"
    VERIFIED = "verified"
    FALSIFIED = "falsified"
    UNKNOWN = "unknown"
    ERROR = "error"
    CORRECT_UNDER_PERTURBATION = "correct_under_perturbation"
    MISCLASSIFIED_UNDER_PERTURBATION = "misclassified_under_perturbation"


# Stable integer encoding for storing verdicts inside a torch tensor.
# Map order is the public contract; do not renumber existing entries.
VERDICT_CODES: dict[RobustnessVerdict, int] = {
    RobustnessVerdict.ATTACK_SUCCEEDED: 1,
    RobustnessVerdict.ATTACK_FAILED: 2,
    RobustnessVerdict.VERIFIED: 3,
    RobustnessVerdict.FALSIFIED: 4,
    RobustnessVerdict.UNKNOWN: 5,
    RobustnessVerdict.ERROR: 6,
    RobustnessVerdict.CORRECT_UNDER_PERTURBATION: 7,
    RobustnessVerdict.MISCLASSIFIED_UNDER_PERTURBATION: 8,
}
VERDICT_FROM_CODE: dict[int, RobustnessVerdict] = {code: v for v, code in VERDICT_CODES.items()}


def encode_verdict(verdict: RobustnessVerdict) -> int:
    return VERDICT_CODES[verdict]


def decode_verdict(code: int) -> RobustnessVerdict:
    try:
        return VERDICT_FROM_CODE[code]
    except KeyError as error:
        raise ValueError(f"Unknown robustness verdict code {code!r}.") from error


class ThreatModel(StrEnum):
    """Adversary capability assumed by the assessor."""

    WHITE_BOX = "white_box"
    BLACK_BOX_SCORE = "black_box_score"
    BLACK_BOX_DECISION = "black_box_decision"
    NOT_APPLICABLE = "not_applicable"


class Objective(StrEnum):
    """Whether the assessor seeks any mis-classification or a specific target class."""

    UNTARGETED = "untargeted"
    TARGETED = "targeted"


class PerturbationNorm(StrEnum):
    """Norm under which the perturbation budget is measured."""

    LINF = "Linf"
    L2 = "L2"
    L1 = "L1"
    L0 = "L0"


@dataclass(frozen=True)
class PerturbationRegion:
    """Base for the region of inputs an assessment explores. Kind-specific subclasses."""


@dataclass(frozen=True)
class PerturbationBudget(PerturbationRegion):
    """Worst-case norm ball an adversary may explore."""

    norm: PerturbationNorm
    epsilon: float | None = None
    step_size: float | None = None
    steps: int | None = None


@dataclass(frozen=True)
class PerturbationDistribution(PerturbationRegion):
    """Average-case perturbation distribution (one ImageNet-C corruption at one severity)."""

    corruption_name: str
    severity: int


@dataclass(frozen=True)
class RobustnessSemantics:
    """Typed contract describing the meaning of a robustness assessment artifact."""

    assessment_kind: AssessmentKind
    threat_model: ThreatModel
    objective: Objective
    families: AbstractSet[str]
    perturbation: PerturbationRegion
    target_classes: Sequence[int] | None = None
    sample_selection: SampleSelection | None = None
    input_spec: InputSpec | None = None
    # RNG-source classification (issue #339). Replaces the old ``stochastic``
    # bool. ``deterministic`` => bit-reproducible; ``global_rng`` => covered by a
    # pinned global seed; ``self_seeded`` => owns a seed param, needs it passed.
    seeding: Seeding = "deterministic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "families", frozenset(self.families))

    @property
    def case(self) -> RobustnessCase:
        """Robustness case this assessment belongs to (derived from ``assessment_kind``)."""
        return case_for(self.assessment_kind)

    @property
    def stochastic(self) -> bool:
        """True when the result is RNG-dependent (derived from ``seeding``)."""
        return self.seeding != "deterministic"


@dataclass(frozen=True)
class RobustnessVisualisationContext:
    """Standard pipeline-controlled metadata provided to robustness visualisers."""

    algorithm: str
    assessment_kind: AssessmentKind
    sample_names: list[str] | None
    show_sample_names: bool


@runtime_checkable
class AssessorAdapter(Protocol):
    """Protocol every assessor adapter must satisfy.

    Mirrors ``raitap.transparency.contracts.ExplainerAdapter`` but for the
    robustness pipeline: ``assess(model, inputs, targets, …)`` instead of
    ``explain(model, inputs, …)``.
    """

    assessment_kind: ClassVar[AssessmentKind]

    def check_backend_compat(self, backend: object) -> None:
        pass

    def assess(
        self,
        model: Module,
        inputs: Tensor,
        targets: Tensor,
        *,
        backend: object | None = None,
        run_dir: str | Path | None = None,
        output_root: str | Path | None = None,
        experiment_name: str | None = None,
        assessor_target: str | None = None,
        assessor_name: str | None = None,
        visualisers: list[ConfiguredRobustnessVisualiser] | None = None,
        **kwargs: Any,
    ) -> RobustnessResult:
        raise NotImplementedError


@dataclass(frozen=True)
class VerificationOutcome:
    """Per-sample result returned by a ``FormalVerificationAssessor``.

    ``counter_example`` is set only when the verifier produced an explicit
    falsification; ``lower_bounds`` / ``upper_bounds`` are per-class logit bounds
    when the verifier exposes them. ``runtime_seconds`` measures the verifier's
    own time-to-decision for this sample.
    """

    verdict: RobustnessVerdict
    counter_example: Tensor | None = None
    lower_bounds: Tensor | None = None
    upper_bounds: Tensor | None = None
    runtime_seconds: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
