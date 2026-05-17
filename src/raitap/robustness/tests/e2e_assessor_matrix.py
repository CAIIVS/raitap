"""Per-algorithm E2E matrix for empirical robustness assessors.

Mirrors :mod:`raitap.transparency.tests.e2e_case_matrix` but kept minimal:
one happy-path case per algorithm, shared tiny model + inputs, smoke-level
shape assertion. Heavier algorithm-specific behaviour tests stay in their
dedicated test files (e.g. ``test_e2e_real_data.py``).

CI budget: each case must finish in under ~5s on the GitHub runners — the
constructor kwargs below are tuned to the lowest iteration count that still
exercises the codepath. Don't bump them globally; if you need more iterations
to validate a specific algorithm's behaviour, add a dedicated test in
``test_e2e_real_data.py`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class AssessorMatrixCase:
    id: str
    family: Literal["torchattacks", "foolbox"]
    algorithm: str
    needs_extra: Literal["torchattacks", "foolbox"]
    constructor_kwargs: dict[str, object] = field(default_factory=dict)
    # Foolbox consumes the budget at call time via ``epsilons=``; torchattacks
    # bakes it into ``constructor_kwargs``. ``call_kwargs`` covers the foolbox
    # side without bloating the torchattacks rows.
    call_kwargs: dict[str, object] = field(default_factory=dict)


# Tiny perturbation budgets / iteration counts everywhere — these tests gate
# "does the algorithm wire up end-to-end" rather than "does it find a real
# adversarial example". Behaviour-sensitivity tests live in test_e2e_real_data.py.
MATRIX_CASES: tuple[AssessorMatrixCase, ...] = (
    # ---------- torchattacks ----------
    AssessorMatrixCase(
        id="torchattacks_fgsm",
        family="torchattacks",
        algorithm="FGSM",
        needs_extra="torchattacks",
        constructor_kwargs={"eps": 0.05},
    ),
    AssessorMatrixCase(
        id="torchattacks_bim",
        family="torchattacks",
        algorithm="BIM",
        needs_extra="torchattacks",
        constructor_kwargs={"eps": 0.05, "alpha": 0.02, "steps": 3},
    ),
    AssessorMatrixCase(
        id="torchattacks_pgd",
        family="torchattacks",
        algorithm="PGD",
        needs_extra="torchattacks",
        constructor_kwargs={"eps": 0.05, "alpha": 0.02, "steps": 3},
    ),
    AssessorMatrixCase(
        id="torchattacks_pgdl2",
        family="torchattacks",
        algorithm="PGDL2",
        needs_extra="torchattacks",
        constructor_kwargs={"eps": 0.5, "alpha": 0.2, "steps": 3},
    ),
    AssessorMatrixCase(
        id="torchattacks_mifgsm",
        family="torchattacks",
        algorithm="MIFGSM",
        needs_extra="torchattacks",
        constructor_kwargs={"eps": 0.05, "alpha": 0.02, "steps": 3, "decay": 1.0},
    ),
    AssessorMatrixCase(
        id="torchattacks_cw",
        family="torchattacks",
        algorithm="CW",
        needs_extra="torchattacks",
        constructor_kwargs={"c": 1.0, "kappa": 0.0, "steps": 5, "lr": 0.01},
    ),
    AssessorMatrixCase(
        id="torchattacks_deepfool",
        family="torchattacks",
        algorithm="DeepFool",
        needs_extra="torchattacks",
        constructor_kwargs={"steps": 5, "overshoot": 0.02},
    ),
    AssessorMatrixCase(
        id="torchattacks_square",
        family="torchattacks",
        algorithm="Square",
        needs_extra="torchattacks",
        constructor_kwargs={"eps": 0.05, "n_queries": 20, "n_restarts": 1},
    ),
    AssessorMatrixCase(
        id="torchattacks_onepixel",
        family="torchattacks",
        algorithm="OnePixel",
        needs_extra="torchattacks",
        constructor_kwargs={"pixels": 1, "steps": 5, "popsize": 4},
    ),
    # AutoAttack runs an ensemble (APGD-CE + APGD-DLR + FAB + Square) and is
    # genuinely slow even with minimal kwargs. Excluded from the matrix —
    # exercised separately if needed via a dedicated long-running e2e test.
    # ---------- foolbox ----------
    AssessorMatrixCase(
        id="foolbox_linf_pgd",
        family="foolbox",
        algorithm="LinfPGD",
        needs_extra="foolbox",
        constructor_kwargs={"steps": 3, "abs_stepsize": 0.02},
        call_kwargs={"eps": 0.05},
    ),
    AssessorMatrixCase(
        id="foolbox_l2_pgd",
        family="foolbox",
        algorithm="L2PGD",
        needs_extra="foolbox",
        constructor_kwargs={"steps": 3, "abs_stepsize": 0.2},
        call_kwargs={"eps": 0.5},
    ),
    AssessorMatrixCase(
        id="foolbox_linf_fast_gradient_attack",
        family="foolbox",
        algorithm="LinfFastGradientAttack",
        needs_extra="foolbox",
        call_kwargs={"eps": 0.05},
    ),
    AssessorMatrixCase(
        id="foolbox_l2_fast_gradient_attack",
        family="foolbox",
        algorithm="L2FastGradientAttack",
        needs_extra="foolbox",
        call_kwargs={"eps": 0.5},
    ),
    AssessorMatrixCase(
        id="foolbox_l2_cw_attack",
        family="foolbox",
        algorithm="L2CarliniWagnerAttack",
        needs_extra="foolbox",
        constructor_kwargs={"steps": 5, "binary_search_steps": 2, "initial_const": 0.1},
        call_kwargs={"eps": 1.0},
    ),
    AssessorMatrixCase(
        id="foolbox_l2_deepfool",
        family="foolbox",
        algorithm="L2DeepFoolAttack",
        needs_extra="foolbox",
        constructor_kwargs={"steps": 5},
        call_kwargs={"eps": 1.0},
    ),
    AssessorMatrixCase(
        id="foolbox_boundary",
        family="foolbox",
        algorithm="BoundaryAttack",
        needs_extra="foolbox",
        constructor_kwargs={"steps": 5},
        call_kwargs={"eps": 1.0},
    ),
)
