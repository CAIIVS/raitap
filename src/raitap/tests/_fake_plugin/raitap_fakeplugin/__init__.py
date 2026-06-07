from raitap import adapters
from raitap.robustness.assessors.base_assessor import AttackInvokeCtx, EmpiricalAttackAssessor
from raitap.robustness.contracts import AssessmentKind, Objective, PerturbationNorm, ThreatModel
from raitap.robustness.semantics import AssessorAlgorithmSpec


@adapters.robustness(
    registry_name="fakeattack",
    algorithm_registry={
        "PGD": AssessorAlgorithmSpec(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families=frozenset({"iterative"}),
        ),
    },
)
class FakeAttackAssessor(EmpiricalAttackAssessor):
    def _default_invoke(self, ctx: AttackInvokeCtx):  # noqa: ANN202
        return ctx.inputs
