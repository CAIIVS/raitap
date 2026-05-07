"""Semantic registries for robustness assessors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class AssessorSemanticsHints:
    """Per-algorithm metadata read by :func:`assessor_semantics`."""

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


_TORCHATTACKS_TARGET_HINTS = ("torchattacks", "TorchattacksAssessor")
_FOOLBOX_TARGET_HINTS = ("foolbox", "FoolboxAssessor")


_TARGET_KWARG_KEYS: frozenset[str] = frozenset(
    {"target_labels", "target_classes", "target_class", "target"}
)


def _registry_for(assessor: object) -> Mapping[str, AssessorSemanticsHints]:
    cls = type(assessor)
    qualified = f"{cls.__module__}.{cls.__name__}".lower()
    name = cls.__name__.lower()
    if any(hint.lower() in qualified or hint.lower() in name for hint in _TORCHATTACKS_TARGET_HINTS):
        return TORCHATTACKS_REGISTRY
    if any(hint.lower() in qualified or hint.lower() in name for hint in _FOOLBOX_TARGET_HINTS):
        return FOOLBOX_REGISTRY
    raise ValueError(
        f"No semantics registry registered for assessor type {cls.__name__!r}. "
        "Add a TORCHATTACKS_REGISTRY / FOOLBOX_REGISTRY entry, or extend "
        "raitap.robustness.semantics with a new framework registry."
    )


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


def _resolve_objective(
    hints: AssessorSemanticsHints, call_kwargs: Mapping[str, Any]
) -> Objective:
    if _extract_target_classes(call_kwargs) is not None:
        return Objective.TARGETED
    return hints.objective


def _resolve_epsilon(
    hints: AssessorSemanticsHints, call_kwargs: Mapping[str, Any]
) -> float | None:
    for key in ("eps", "epsilon"):
        if key in call_kwargs and call_kwargs[key] is not None:
            value = call_kwargs[key]
            if isinstance(value, (list, tuple)):
                # multi-epsilon sweeps fall outside the rework scope; the foolbox
                # adapter validates this earlier with a clearer message.
                return None
            return float(value)
    if "epsilons" in call_kwargs and call_kwargs["epsilons"] is not None:
        value = call_kwargs["epsilons"]
        if isinstance(value, (int, float)):
            return float(value)
        # list / tuple ⇒ sweep, see above.
        return None
    return hints.default_epsilon


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
    """Build a :class:`RobustnessSemantics` from the configured assessor and its kwargs."""
    del targets  # reserved for future per-sample target metadata; not used yet.
    hints = hints_for_assessor(assessor)
    objective = _resolve_objective(hints, call_kwargs)
    epsilon = _resolve_epsilon(hints, call_kwargs)
    step_size = call_kwargs.get("alpha")
    if step_size is None:
        step_size = call_kwargs.get("step_size")
    steps = call_kwargs.get("steps")
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
