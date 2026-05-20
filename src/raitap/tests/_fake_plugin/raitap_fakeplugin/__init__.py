from raitap import adapters
from raitap.robustness.assessors.base_assessor import EmpiricalAttackAssessor
from raitap.robustness.contracts import MethodKind, Objective, PerturbationNorm, ThreatModel
from raitap.robustness.semantics import AssessorSemanticsHints


@adapters.robustness(
    registry_name="fakeattack",
    algorithm_registry={
        "PGD": AssessorSemanticsHints(
            MethodKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families=frozenset({"iterative"}),
        ),
    },
)
class FakeAttackAssessor(EmpiricalAttackAssessor):
    def generate_adversarial(self, model, inputs, targets, *, backend=None, **kw):  # noqa: ANN001, ANN201
        return inputs
