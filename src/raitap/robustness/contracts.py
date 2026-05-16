"""
Shared robustness contracts: method-kind taxonomy, threat model, and verdict typing.

The robustness module distinguishes two fundamentally different ways to assess a
model's robustness against perturbations:

* ``MethodKind.EMPIRICAL_ATTACK`` — try to find an adversarial example within a
  perturbation budget (torchattacks, foolbox, …). A "non-attack" outcome does
  *not* prove robustness; it just means the configured attack failed.
* ``MethodKind.FORMAL_VERIFICATION`` — prove (or refute) that no adversarial
  example exists within the budget (auto_LiRPA, alpha-beta-CROWN, ...). Outcomes are
  verified / falsified / unknown.

A single ``RobustnessSemantics`` carries this distinction so that downstream
result handling, reporting, and visualiser-compatibility checks can branch on
``method_kind`` instead of duck-typing.
"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import Any, ClassVar, Protocol, runtime_checkable

from raitap.transparency.contracts import InputSpec, SampleSelection  # noqa: TC001

ConfiguredRobustnessVisualiser = Any
RobustnessResult = Any
Module = Any
Tensor = Any


class MethodKind(StrEnum):
    """High-level taxonomy distinguishing empirical attacks from formal verification."""

    EMPIRICAL_ATTACK = "empirical_attack"
    FORMAL_VERIFICATION = "formal_verification"


class RobustnessVerdict(StrEnum):
    """Per-sample assessment outcome.

    Empirical assessors emit ``ATTACKED`` / ``NOT_ATTACKED`` (the latter does NOT
    prove robustness — it only means the configured attack failed).
    Formal verification assessors emit ``VERIFIED`` / ``FALSIFIED`` / ``UNKNOWN``
    (and ``ERROR`` for per-sample crashes / timeouts).
    """

    ATTACKED = "attacked"
    NOT_ATTACKED = "not_attacked"
    VERIFIED = "verified"
    FALSIFIED = "falsified"
    UNKNOWN = "unknown"
    ERROR = "error"


# Stable integer encoding for storing verdicts inside a torch tensor.
# Map order is the public contract; do not renumber existing entries.
VERDICT_CODES: dict[RobustnessVerdict, int] = {
    RobustnessVerdict.ATTACKED: 1,
    RobustnessVerdict.NOT_ATTACKED: 2,
    RobustnessVerdict.VERIFIED: 3,
    RobustnessVerdict.FALSIFIED: 4,
    RobustnessVerdict.UNKNOWN: 5,
    RobustnessVerdict.ERROR: 6,
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
class PerturbationBudget:
    """Region of inputs an adversary is allowed to explore."""

    norm: PerturbationNorm
    epsilon: float | None = None
    step_size: float | None = None
    steps: int | None = None


@dataclass(frozen=True)
class RobustnessSemantics:
    """Typed contract describing the meaning of a robustness assessment artifact."""

    method_kind: MethodKind
    threat_model: ThreatModel
    objective: Objective
    families: frozenset[str]
    budget: PerturbationBudget
    target_classes: Sequence[int] | None = None
    sample_selection: SampleSelection | None = None
    input_spec: InputSpec | None = None


@dataclass(frozen=True)
class RobustnessVisualisationContext:
    """Standard pipeline-controlled metadata provided to robustness visualisers."""

    algorithm: str
    method_kind: MethodKind
    sample_names: list[str] | None
    show_sample_names: bool


@runtime_checkable
class AssessorAdapter(Protocol):
    """Protocol every assessor adapter must satisfy.

    Mirrors ``raitap.transparency.contracts.ExplainerAdapter`` but for the
    robustness pipeline: ``assess(model, inputs, targets, …)`` instead of
    ``explain(model, inputs, …)``.
    """

    method_kind: ClassVar[MethodKind]

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
