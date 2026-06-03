"""Foolbox adapter for RAITAP robustness assessments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.robustness.assessors.registration import robustness_adapter
from raitap.types import Capability
from raitap.utils.lazy import lazy_import

from ..contracts import AssessmentKind, Objective, PerturbationNorm, ThreatModel
from ..semantics import AssessorSemanticsHints
from .base_assessor import EmpiricalAttackAssessor, _prepare_inputs_for_forward

if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    torch = lazy_import("torch")


@robustness_adapter(
    registry_name="foolbox",
    library="foolbox",
    budget_kwarg_source="call_kwargs",
    algorithm_registry={
        "LinfPGD": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families=frozenset({"gradient_sign", "iterative"}),
            requires=frozenset({Capability.AUTOGRAD}),
        ),
        "L2PGD": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L2,
            families=frozenset({"gradient_sign", "iterative"}),
            requires=frozenset({Capability.AUTOGRAD}),
        ),
        "LinfFastGradientAttack": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families=frozenset({"gradient_sign"}),
            requires=frozenset({Capability.AUTOGRAD}),
        ),
        "L2FastGradientAttack": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L2,
            families=frozenset({"gradient_sign"}),
            requires=frozenset({Capability.AUTOGRAD}),
        ),
        "L2CarliniWagnerAttack": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L2,
            families=frozenset({"optimization"}),
            requires=frozenset({Capability.AUTOGRAD}),
        ),
        "L2DeepFoolAttack": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L2,
            families=frozenset({"optimization"}),
            requires=frozenset({Capability.AUTOGRAD}),
        ),
        "BoundaryAttack": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.BLACK_BOX_DECISION,
            Objective.UNTARGETED,
            PerturbationNorm.L2,
            families=frozenset({"decision_boundary", "query_efficient"}),
            # requires=AUTOGRAD preserves current gating; conceptually black-box,
            # could be relaxed later.
            requires=frozenset({Capability.AUTOGRAD}),
        ),
    },
)
class FoolboxAssessor(EmpiricalAttackAssessor):
    """Single wrapper for foolbox attack classes.

    Foolbox consumes the perturbation budget at *call time* (``attack(fmodel,
    inputs, targets, epsilons=...)``), so the YAML budget keys belong under
    ``call:``; we set ``budget_kwarg_source = "call_kwargs"`` so semantics
    metadata reflects that.

    Multi-epsilon sweeps (passing a list to ``epsilons`` so foolbox returns a
    per-eps list of tensors) are intentionally **not** supported in this adapter
    — they would change the result tensor shape across configurations and break
    the uniform ``RobustnessResult`` contract.
    A future ``MultiEpsilonAssessor`` will own that surface.
    """

    def __init__(
        self,
        algorithm: str,
        *,
        bounds: tuple[float, float] = (0.0, 1.0),
        preprocessing: dict[str, Any] | None = None,
        **init_kwargs: Any,
    ) -> None:
        self.algorithm = algorithm
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.preprocessing = dict(preprocessing) if preprocessing else None
        self.init_kwargs = dict(init_kwargs)
        self._last_success: torch.Tensor | None = None

    def generate_adversarial(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        backend: object | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        foolbox = self._lazy_import()
        attacks_module = foolbox.attacks
        try:
            attack_class = getattr(attacks_module, self.algorithm)
        except AttributeError as error:
            raise ValueError(
                f"{self.algorithm!r} is not a valid foolbox.attacks class name."
            ) from error

        attack = attack_class(**self.init_kwargs)

        eps = self._extract_scalar_eps(kwargs)
        criterion = self._build_criterion(foolbox, kwargs, targets)

        # Route device placement through the backend so the foolbox model wrapper sees
        # the same device the rest of the pipeline forwards through. Defensive contiguity
        # because RAITAP's image loader produces non-contiguous NCHW via HWC->CHW.
        inputs_dev = _prepare_inputs_for_forward(inputs, model=model, backend=backend).contiguous()
        targets_dev = (
            criterion
            if criterion is not None
            else _prepare_inputs_for_forward(targets, model=model, backend=backend).contiguous()
        )

        fmodel = foolbox.PyTorchModel(  # pyright: ignore[reportPrivateImportUsage]
            model,
            bounds=self.bounds,
            preprocessing=self.preprocessing,
        )

        with self._rethrow():
            raw, clipped, success = attack(fmodel, inputs_dev, targets_dev, epsilons=eps)
        del raw  # unclipped — we keep the clipped tensor only
        if isinstance(clipped, list):
            raise TypeError(
                "FoolboxAssessor received a list of perturbed tensors, which means a "
                "multi-epsilon sweep was requested. Pass a scalar `eps` / `epsilons` "
                "value; multi-epsilon support is intentionally out of scope here."
            )
        if isinstance(success, torch.Tensor):
            self._last_success = success.detach().cpu()
        return clipped.detach()

    @staticmethod
    def _extract_scalar_eps(kwargs: dict[str, Any]) -> float:
        for key in ("eps", "epsilon", "epsilons"):
            if key in kwargs and kwargs[key] is not None:
                value = kwargs.pop(key)
                if isinstance(value, (list, tuple)):
                    raise TypeError(
                        "FoolboxAssessor expects a scalar epsilon. "
                        f"Got {type(value).__name__} under key {key!r}; multi-epsilon "
                        "sweeps are out of scope for this adapter."
                    )
                return float(value)
        raise ValueError(
            "FoolboxAssessor requires `call.eps` (or `call.epsilon` / `call.epsilons`) to be set."
        )

    @staticmethod
    def _build_criterion(
        foolbox_module: Any,
        kwargs: dict[str, Any],
        targets: torch.Tensor,
    ) -> Any:
        target_labels = kwargs.pop("target_labels", None)
        target_classes = kwargs.pop("target_classes", None)
        if target_labels is None and target_classes is None:
            return None
        criteria = foolbox_module.criteria
        wanted = target_labels if target_labels is not None else target_classes
        if not isinstance(wanted, torch.Tensor):
            if isinstance(wanted, int):
                wanted = torch.full_like(targets, fill_value=int(wanted), dtype=torch.long)
            else:
                wanted = torch.tensor(list(wanted), dtype=torch.long)
        return criteria.TargetedMisclassification(wanted)
