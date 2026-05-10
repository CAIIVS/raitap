"""Semantic registries for robustness assessors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from raitap.transparency.contracts import InputSpec, SampleSelection
from raitap.transparency.semantics import infer_input_spec

from .contracts import (
    MethodKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessSemantics,
    ThreatModel,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class AssessorSemanticsHints:
    """Per-algorithm metadata read by ``assessor_semantics``."""

    method_kind: MethodKind
    threat_model: ThreatModel
    objective: Objective
    norm: PerturbationNorm
    families: frozenset[str]
    default_epsilon: float | None = None


# Mapping of torchattacks algorithm names to their semantic hints.
TORCHATTACKS_REGISTRY: Mapping[str, AssessorSemanticsHints] = {
    "FGSM": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.LINF,
        families=frozenset({"gradient_sign"}),
    ),
    "BIM": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.LINF,
        families=frozenset({"gradient_sign", "iterative"}),
    ),
    "PGD": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.LINF,
        families=frozenset({"gradient_sign", "iterative"}),
    ),
    "PGDL2": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.L2,
        families=frozenset({"gradient_sign", "iterative"}),
    ),
    "MIFGSM": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.LINF,
        families=frozenset({"gradient_sign", "iterative", "momentum"}),
    ),
    "CW": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.L2,
        families=frozenset({"optimization"}),
    ),
    "DeepFool": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.L2,
        families=frozenset({"optimization"}),
    ),
    "AutoAttack": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.LINF,
        families=frozenset({"ensemble", "auto"}),
    ),
    "Square": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.BLACK_BOX_SCORE,
        Objective.UNTARGETED,
        PerturbationNorm.LINF,
        families=frozenset({"score_based"}),
    ),
    "OnePixel": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.BLACK_BOX_SCORE,
        Objective.UNTARGETED,
        PerturbationNorm.L0,
        families=frozenset({"score_based", "evolutionary"}),
    ),
}

# Mapping of foolbox attack class names to their semantic hints.
FOOLBOX_REGISTRY: Mapping[str, AssessorSemanticsHints] = {
    "LinfPGD": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.LINF,
        families=frozenset({"gradient_sign", "iterative"}),
    ),
    "L2PGD": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.L2,
        families=frozenset({"gradient_sign", "iterative"}),
    ),
    "LinfFastGradientAttack": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.LINF,
        families=frozenset({"gradient_sign"}),
    ),
    "L2FastGradientAttack": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.L2,
        families=frozenset({"gradient_sign"}),
    ),
    "L2CarliniWagnerAttack": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.L2,
        families=frozenset({"optimization"}),
    ),
    "L2DeepFoolAttack": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.WHITE_BOX,
        Objective.UNTARGETED,
        PerturbationNorm.L2,
        families=frozenset({"optimization"}),
    ),
    "BoundaryAttack": AssessorSemanticsHints(
        MethodKind.EMPIRICAL_ATTACK,
        ThreatModel.BLACK_BOX_DECISION,
        Objective.UNTARGETED,
        PerturbationNorm.L2,
        families=frozenset({"decision_boundary", "query_efficient"}),
    ),
}


_TARGET_KWARG_KEYS: frozenset[str] = frozenset(
    {"target_labels", "target_classes", "target_class", "target"}
)


def _registry_for(assessor: object) -> Mapping[str, AssessorSemanticsHints]:
    """Read the assessor's own ``algorithm_registry`` ClassVar.

    Each adapter declares its own registry on the class so semantics doesn't
    have to branch on framework names. Adding a new framework = subclass
    ``BaseAssessor`` and set the ClassVar; no edits here required.
    """
    registry = getattr(type(assessor), "algorithm_registry", None)
    if registry is None:
        raise ValueError(
            f"Assessor type {type(assessor).__name__!r} does not declare an "
            "``algorithm_registry`` ClassVar. Each assessor adapter must declare "
            "its supported algorithms (e.g. set "
            "``algorithm_registry: ClassVar[Mapping[str, AssessorSemanticsHints]] = ...`` "
            "in the subclass)."
        )
    return registry


def hints_for_assessor(assessor: object) -> AssessorSemanticsHints:
    """Resolve the registry hints for a configured assessor."""
    algorithm = str(getattr(assessor, "algorithm", ""))
    if not algorithm:
        raise ValueError(
            f"Assessor {type(assessor).__name__!r} has no ``algorithm`` attribute. "
            "Set the YAML ``algorithm:`` field (e.g. ``algorithm: PGD``)."
        )
    registry = _registry_for(assessor)
    try:
        return registry[algorithm]
    except KeyError as error:
        valid = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown algorithm {algorithm!r} for assessor {type(assessor).__name__!r}. "
            f"Known algorithms: {valid}."
        ) from error


def _extract_target_classes(call_kwargs: Mapping[str, Any]) -> Sequence[int] | None:
    for key in _TARGET_KWARG_KEYS:
        if key not in call_kwargs:
            continue
        value = call_kwargs[key]
        if value is None:
            continue
        if isinstance(value, int):
            return (int(value),)
        if isinstance(value, (list, tuple)):
            return tuple(int(item) for item in value)
        # tensors are detected structurally to avoid a hard torch dependency at import time
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            collected = tolist()
            if isinstance(collected, list):
                return tuple(int(item) for item in collected)
            if isinstance(collected, int):
                return (int(collected),)
    return None


def _resolve_objective(hints: AssessorSemanticsHints, call_kwargs: Mapping[str, Any]) -> Objective:
    if _extract_target_classes(call_kwargs) is not None:
        return Objective.TARGETED
    return hints.objective


def _resolve_epsilon(
    hints: AssessorSemanticsHints,
    source: Mapping[str, Any],
) -> float | None:
    for key in ("eps", "epsilon"):
        if key in source and source[key] is not None:
            value = source[key]
            if isinstance(value, (list, tuple)):
                # multi-epsilon sweeps fall outside the rework scope; the foolbox
                # adapter validates this earlier with a clearer message.
                return None
            return float(value)
    if "epsilons" in source and source["epsilons"] is not None:
        value = source["epsilons"]
        if isinstance(value, (int, float)):
            return float(value)
        return None  # list / tuple => sweep, see above.
    return hints.default_epsilon


_BUDGET_KEY_GROUPS = (
    ("eps", "epsilon", "epsilons"),
    ("alpha", "step_size"),
    ("steps",),
)


def _warn_misplaced_budget_keys(
    *,
    assessor: object,
    authoritative: str,
    other_source: Mapping[str, Any],
) -> None:
    """Warn when budget keys appear in the source the adapter doesn't read."""
    misplaced = sorted(
        {key for group in _BUDGET_KEY_GROUPS for key in group if key in other_source}
    )
    if not misplaced:
        return
    other_label = "call_kwargs" if authoritative == "init_kwargs" else "init_kwargs"
    other_yaml = "call:" if other_label == "call_kwargs" else "constructor:"
    auth_yaml = "constructor:" if authoritative == "init_kwargs" else "call:"

    from raitap import raitap_log

    raitap_log.warn(
        f"Assessor {type(assessor).__name__} reads budget kwargs from "
        f"{authoritative} (YAML {auth_yaml}) but found {misplaced} under "
        f"{other_label} (YAML {other_yaml}). Those values are ignored by the "
        f"adapter; move them to {auth_yaml} for the configured budget to take effect.",
    )


def assessor_semantics(
    assessor: object,
    *,
    call_kwargs: Mapping[str, Any],
    raitap_kwargs: Mapping[str, Any],
    inputs: object,
    targets: object,
    sample_ids: list[str] | None = None,
    sample_names: list[str] | None = None,
) -> RobustnessSemantics:
    """Build a ``RobustnessSemantics`` from the configured assessor and its kwargs.

    Budget fields (``epsilon``, ``step_size``, ``steps``) live in only one of the
    two YAML blocks per framework, governed by the assessor's
    ``budget_kwarg_source`` ClassVar:

    * torchattacks: ``"init_kwargs"`` (constructor-time, since the adapter does
      ``attack_class(model, **init_kwargs)`` once and never forwards per-call
      budget keys).
    * foolbox: ``"call_kwargs"`` (foolbox attacks consume ``epsilons=...`` at
      ``attack(...)`` time).

    The reported ``RobustnessSemantics.budget`` therefore matches what the
    adapter actually executed. Misplaced budget keys in the non-authoritative
    source emit a warning so the user can correct the YAML.
    """
    del targets  # reserved for future per-sample target metadata; not used yet.
    hints = hints_for_assessor(assessor)
    init_kwargs: Mapping[str, Any] = getattr(assessor, "init_kwargs", {}) or {}
    authoritative_label = str(getattr(assessor, "budget_kwarg_source", "init_kwargs"))
    if authoritative_label == "call_kwargs":
        budget_source: Mapping[str, Any] = call_kwargs
        other_source: Mapping[str, Any] = init_kwargs
    else:
        budget_source = init_kwargs
        other_source = call_kwargs
    _warn_misplaced_budget_keys(
        assessor=assessor, authoritative=authoritative_label, other_source=other_source
    )
    objective = _resolve_objective(hints, call_kwargs)
    epsilon = _resolve_epsilon(hints, budget_source)
    step_size = budget_source.get("alpha")
    if step_size is None:
        step_size = budget_source.get("step_size")
    steps = budget_source.get("steps")
    budget = PerturbationBudget(
        norm=hints.norm,
        epsilon=float(epsilon) if epsilon is not None else None,
        step_size=float(step_size) if step_size is not None else None,
        steps=int(steps) if steps is not None else None,
    )
    input_spec: InputSpec | None
    try:
        input_spec = infer_input_spec(
            inputs,
            input_metadata=raitap_kwargs.get("input_metadata"),
        )
    except Exception:  # pragma: no cover — input_spec is best-effort here
        input_spec = None
    sample_selection = SampleSelection(
        sample_ids=sample_ids,
        sample_display_names=sample_names,
    )
    return RobustnessSemantics(
        method_kind=hints.method_kind,
        threat_model=hints.threat_model,
        objective=objective,
        families=hints.families,
        budget=budget,
        target_classes=_extract_target_classes(call_kwargs),
        sample_selection=sample_selection,
        input_spec=input_spec,
    )
