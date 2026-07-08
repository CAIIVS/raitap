"""Runnable reference plugin for RAITAP's entry-point adapter system.

Registers one trivial robustness assessor, ``identity_attack``, that returns
the inputs unmodified (no real attack logic, no third-party dependency) — it
exists purely to demonstrate the wiring: entry-point discovery, the
``@adapters.robustness`` decorator, and selecting the result with
``use: identity_attack`` in a RAITAP config.

See ``README.md`` in this directory for install + usage instructions, and
``docs/contributor/writing-a-plugin.md`` in the RAITAP repo for the full guide.
"""

from __future__ import annotations

from raitap import adapters
from raitap.robustness.assessors.base_assessor import AttackInvokeCtx, EmpiricalAttackAssessor
from raitap.robustness.contracts import AssessmentKind, Objective, PerturbationNorm, ThreatModel
from raitap.robustness.semantics import AssessorAlgorithmSpec


@adapters.robustness(
    registry_name="identity_attack",
    algorithm_registry={
        "identity": AssessorAlgorithmSpec(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families=frozenset({"identity"}),
        ),
    },
)
class IdentityAttackAssessor(EmpiricalAttackAssessor):
    """No-op "attack": returns the inputs unchanged.

    Not a real robustness check — a minimal, dependency-free adapter body so
    this plugin installs and runs with nothing beyond RAITAP itself.
    """

    def __init__(self, algorithm: str, **init_kwargs) -> None:
        self.algorithm = algorithm
        self.init_kwargs = init_kwargs

    def _default_invoke(self, ctx: AttackInvokeCtx):  # noqa: ANN202
        return ctx.inputs
