"""Foolbox adapter for RAITAP robustness assessments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from raitap.robustness.assessors.registration import robustness_adapter
from raitap.types import Capability
from raitap.utils.lazy import lazy_import

from ..contracts import AssessmentKind, Objective, PerturbationNorm, ThreatModel
from ..semantics import AssessorSemanticsHints
from .base_assessor import AttackInvokeCtx, EmpiricalAttackAssessor, _prepare_inputs_for_forward

if TYPE_CHECKING:
    import torch
else:
    torch = lazy_import("torch")


def _dataset_attack_invoker(ctx: AttackInvokeCtx) -> torch.Tensor:
    """foolbox DatasetAttack needs .feed() to load a sample pool before running (#266)."""
    foolbox = ctx.library
    a = cast("FoolboxAssessor", ctx.assessor)
    kwargs = dict(ctx.call_kwargs)
    # DatasetAttack is FlexibleDistance: it raises at call-time without a distance.
    # Default to L2 (nearest-sample) when the user didn't pin one in the constructor.
    init_kwargs = dict(a.init_kwargs)
    init_kwargs.setdefault("distance", foolbox.distances.l2)
    attack = foolbox.attacks.DatasetAttack(**init_kwargs)
    inputs_dev = _prepare_inputs_for_forward(
        ctx.inputs, model=ctx.model, backend=ctx.backend
    ).contiguous()
    fmodel = foolbox.PyTorchModel(  # pyright: ignore[reportPrivateImportUsage]
        ctx.model, bounds=a.bounds, preprocessing=a.preprocessing
    )
    attack.feed(fmodel, inputs_dev)  # feed(model, inputs) -> builds the sample pool
    eps = a._extract_scalar_eps(kwargs)
    # Honor targeted mode the same way the default path does (build a criterion
    # from target_labels/target_classes); fall back to the clean labels otherwise.
    criterion = a._build_criterion(foolbox, kwargs, ctx.targets)
    target_arg = (
        criterion
        if criterion is not None
        else _prepare_inputs_for_forward(
            ctx.targets, model=ctx.model, backend=ctx.backend
        ).contiguous()
    )
    with a._rethrow():
        _raw, clipped, success = attack(fmodel, inputs_dev, target_arg, epsilons=eps)
    if isinstance(success, torch.Tensor):
        a._last_success = success.detach().cpu()
    return clipped.detach()


# Hint factory: every foolbox attack is gradient/query-driven; AUTOGRAD preserves the
# adapter's gating (see BoundaryAttack note). Only norm / threat / families / stochastic
# vary per entry, all verified against installed foolbox 3.3.4 ctor defaults + source.
def _hint(
    threat: ThreatModel,
    norm: PerturbationNorm | None,
    families: set[str],
    *,
    stochastic: bool = False,
    objective: Objective = Objective.UNTARGETED,
    invoker: object = None,
) -> AssessorSemanticsHints:
    return AssessorSemanticsHints(
        AssessmentKind.EMPIRICAL_ATTACK,
        threat,
        objective,
        norm,
        families=families,
        requires={Capability.AUTOGRAD},
        stochastic=stochastic,
        invoker=invoker,
    )


_WB = ThreatModel.WHITE_BOX
_BBD = ThreatModel.BLACK_BOX_DECISION
_BBS = ThreatModel.BLACK_BOX_SCORE
_LINF = PerturbationNorm.LINF
_L2 = PerturbationNorm.L2
_L1 = PerturbationNorm.L1
_L0 = PerturbationNorm.L0


# Curated rewrites for opaque foolbox errors. Matched against str(exc); first hit wins.
# See :func:`raitap.utils.errors.rethrow`.
_FOOLBOX_ERROR_MESSAGES: dict[str, str] = {
    # FlexibleDistance attacks (GaussianBlurAttack, BinarySearchContrastReductionAttack,
    # LinearSearchContrastReductionAttack, InversionAttack, LinearSearchBlendedUniformNoiseAttack)
    # raise this ValueError at call-time when no distance was passed to the constructor.
    r"unknown distance, please pass `distance` to the attack initializer": (
        "This foolbox attack is FlexibleDistance: it needs an explicit distance norm. "
        "Set `constructor.distance` in the YAML (e.g. `distance: l2`) for this attack."
    ),
    # VirtualAdversarialAttack raises a TypeError at construction time when the
    # required `steps` argument is omitted.
    r"VirtualAdversarialAttack\.__init__\(\) missing .* argument.*['\"]steps['\"]": (
        "VirtualAdversarialAttack requires `steps` (number of power iterations). "
        "Set `constructor.steps` in the YAML (e.g. `steps: 10`) for this attack."
    ),
}


@robustness_adapter(
    registry_name="foolbox",
    library="foolbox",
    budget_kwarg_source="call_kwargs",
    error_patterns=_FOOLBOX_ERROR_MESSAGES,
    algorithm_registry={
        # --- Projected Gradient Descent (random_start=True default -> stochastic) ---
        # LinfPGD collapses aliases {LinfProjectedGradientDescentAttack, PGD}.
        "LinfPGD": _hint(_WB, _LINF, {"gradient_sign", "iterative"}, stochastic=True),
        "L2PGD": _hint(  # aliases {L2ProjectedGradientDescentAttack}
            _WB, _L2, {"gradient_sign", "iterative"}, stochastic=True
        ),
        "L1PGD": _hint(  # aliases {L1ProjectedGradientDescentAttack}
            _WB, _L1, {"gradient_sign", "iterative"}, stochastic=True
        ),
        # Adam-PGD variants (random_start=True default -> stochastic).
        "LinfAdamPGD": _hint(  # aliases {AdamPGD, LinfAdamProjectedGradientDescentAttack}
            _WB, _LINF, {"gradient_sign", "iterative", "adam"}, stochastic=True
        ),
        "L2AdamPGD": _hint(  # aliases {L2AdamProjectedGradientDescentAttack}
            _WB, _L2, {"gradient_sign", "iterative", "adam"}, stochastic=True
        ),
        "L1AdamPGD": _hint(  # aliases {L1AdamProjectedGradientDescentAttack}
            _WB, _L1, {"gradient_sign", "iterative", "adam"}, stochastic=True
        ),
        # --- Fast Gradient (single step, random_start=False -> deterministic) ---
        "LinfFastGradientAttack": _hint(  # aliases {FGSM}
            _WB, _LINF, {"gradient_sign"}
        ),
        "L2FastGradientAttack": _hint(_WB, _L2, {"gradient_sign"}),  # aliases {FGM}
        "L1FastGradientAttack": _hint(_WB, _L1, {"gradient_sign"}),
        # --- Basic Iterative (random_start=False default -> deterministic) ---
        "LinfBasicIterativeAttack": _hint(_WB, _LINF, {"gradient_sign", "iterative"}),
        "L2BasicIterativeAttack": _hint(_WB, _L2, {"gradient_sign", "iterative"}),
        "L1BasicIterativeAttack": _hint(_WB, _L1, {"gradient_sign", "iterative"}),
        "LinfAdamBasicIterativeAttack": _hint(_WB, _LINF, {"gradient_sign", "iterative", "adam"}),
        "L2AdamBasicIterativeAttack": _hint(_WB, _L2, {"gradient_sign", "iterative", "adam"}),
        "L1AdamBasicIterativeAttack": _hint(_WB, _L1, {"gradient_sign", "iterative", "adam"}),
        # --- Momentum Iterative / MI-FGSM (random_start=False -> deterministic) ---
        # LinfMomentumIterativeFastGradientMethod collapses alias {MIFGSM}.
        "LinfMomentumIterativeFastGradientMethod": _hint(
            _WB, _LINF, {"gradient_sign", "iterative", "momentum"}
        ),
        "L2MomentumIterativeFastGradientMethod": _hint(
            _WB, _L2, {"gradient_sign", "iterative", "momentum"}
        ),
        "L1MomentumIterativeFastGradientMethod": _hint(
            _WB, _L1, {"gradient_sign", "iterative", "momentum"}
        ),
        # --- Sparse L1 descent (random_start=False -> deterministic) ---
        "SparseL1DescentAttack": _hint(_WB, _L1, {"gradient_sign", "iterative", "sparse"}),
        # --- Carlini-Wagner / EAD / DeepFool (optimization, deterministic) ---
        "L2CarliniWagnerAttack": _hint(_WB, _L2, {"optimization"}),
        "EADAttack": _hint(_WB, _L1, {"optimization"}),  # distance attr = L1
        "L2DeepFoolAttack": _hint(_WB, _L2, {"optimization"}),
        "LinfDeepFoolAttack": _hint(_WB, _LINF, {"optimization"}),
        # --- DDN / NewtonFool (optimization, deterministic; distance attr = L2) ---
        "DDNAttack": _hint(_WB, _L2, {"optimization"}),
        "NewtonFoolAttack": _hint(_WB, _L2, {"optimization"}),
        # --- Virtual Adversarial Training attack (#266; user supplies steps= via init) ---
        # Source (foolbox 3.3.4): `distance = l2`; random init dir via `ep.normal` ->
        # stochastic; power-iteration on `value_and_grad` (KL-div gradient) -> WHITE_BOX
        # + AUTOGRAD. ctor sig is (steps, xi=1e-06): steps is required (no default).
        "VirtualAdversarialAttack": _hint(
            _WB, _L2, {"optimization", "virtual_adversarial"}, stochastic=True
        ),
        # --- Fast Minimum-Norm (deterministic; norm from class prefix) ---
        "L0FMNAttack": _hint(_WB, _L0, {"optimization", "minimum_norm"}),
        "L1FMNAttack": _hint(_WB, _L1, {"optimization", "minimum_norm"}),
        "L2FMNAttack": _hint(_WB, _L2, {"optimization", "minimum_norm"}),
        # NB: class name is LInfFMNAttack (capital I), not LinfFMNAttack.
        "LInfFMNAttack": _hint(_WB, _LINF, {"optimization", "minimum_norm"}),
        # --- Brendel-Bethge (gradient-based; stochastic init -> stochastic) ---
        "L0BrendelBethgeAttack": _hint(_WB, _L0, {"optimization", "minimum_norm"}, stochastic=True),
        "L1BrendelBethgeAttack": _hint(_WB, _L1, {"optimization", "minimum_norm"}, stochastic=True),
        "L2BrendelBethgeAttack": _hint(_WB, _L2, {"optimization", "minimum_norm"}, stochastic=True),
        # NB: class name is LinfinityBrendelBethgeAttack (Linfinity prefix).
        "LinfinityBrendelBethgeAttack": _hint(
            _WB, _LINF, {"optimization", "minimum_norm"}, stochastic=True
        ),
        # --- Additive-noise score attacks (sample noise -> stochastic; BBS) ---
        "L2AdditiveGaussianNoiseAttack": _hint(_BBS, _L2, {"noise"}, stochastic=True),
        "L2AdditiveUniformNoiseAttack": _hint(_BBS, _L2, {"noise"}, stochastic=True),
        "LinfAdditiveUniformNoiseAttack": _hint(_BBS, _LINF, {"noise"}, stochastic=True),
        "L2RepeatedAdditiveGaussianNoiseAttack": _hint(_BBS, _L2, {"noise"}, stochastic=True),
        "L2RepeatedAdditiveUniformNoiseAttack": _hint(_BBS, _L2, {"noise"}, stochastic=True),
        "LinfRepeatedAdditiveUniformNoiseAttack": _hint(_BBS, _LINF, {"noise"}, stochastic=True),
        "L2ClippingAwareAdditiveGaussianNoiseAttack": _hint(_BBS, _L2, {"noise"}, stochastic=True),
        "L2ClippingAwareAdditiveUniformNoiseAttack": _hint(_BBS, _L2, {"noise"}, stochastic=True),
        "L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack": _hint(
            _BBS, _L2, {"noise"}, stochastic=True
        ),
        "L2ClippingAwareRepeatedAdditiveUniformNoiseAttack": _hint(
            _BBS, _L2, {"noise"}, stochastic=True
        ),
        # --- Contrast / blur / inversion score attacks (deterministic; BBS) ---
        # L2ContrastReductionAttack: FixedEpsilon, distance attr = L2, deterministic.
        "L2ContrastReductionAttack": _hint(_BBS, _L2, {"noise", "contrast"}),
        # FlexibleDistance attacks: ctor takes distance=None and RAISES at call-time
        # without it (verified). Register with a conventional norm; the user MUST set
        # `distance=` (foolbox.distances.l2/l1/...) in the constructor for these to run.
        "BinarySearchContrastReductionAttack": _hint(_BBS, _L2, {"noise", "contrast"}),
        "LinearSearchContrastReductionAttack": _hint(_BBS, _L2, {"noise", "contrast"}),
        "GaussianBlurAttack": _hint(_BBS, _L2, {"noise", "blur"}),
        "InversionAttack": _hint(_BBS, _L2, {"noise", "inversion"}),
        # LinearSearchBlended: random directions -> stochastic.
        "LinearSearchBlendedUniformNoiseAttack": _hint(
            _BBS, _L2, {"noise", "blended"}, stochastic=True
        ),
        # --- Decision-based black-box attacks ---
        # BoundaryAttack: gaussian random-walk + stochastic init -> stochastic.
        "BoundaryAttack": _hint(
            _BBD, _L2, {"decision_boundary", "query_efficient"}, stochastic=True
        ),
        # HopSkipJump: random init + stochastic gradient estimate -> stochastic. constraint='l2'.
        "HopSkipJumpAttack": _hint(
            _BBD, _L2, {"decision_boundary", "query_efficient"}, stochastic=True
        ),
        # SaltAndPepper: random salt/pepper sampling -> stochastic. distance attr = L2;
        # operates by sparsifying pixels (L0-flavoured) but foolbox reports it under L2.
        "SaltAndPepperNoiseAttack": _hint(
            _BBD, _L2, {"decision_boundary", "noise"}, stochastic=True
        ),
        # GenAttack: genetic/population search -> stochastic. distance attr = Linf.
        # Consumes full model logits (continuous C&W-style fitness margin) -> BLACK_BOX_SCORE.
        # Targeted-capable but defaults untargeted.
        "GenAttack": _hint(_BBS, _LINF, {"score_based", "genetic"}, stochastic=True),
        # --- DatasetAttack: needs .feed() before running -> custom invoker (#266) ---
        # Pulls adversarials from a fed sample pool; conceptually L2 nearest-sample.
        "DatasetAttack": _hint(
            _BBD,
            _L2,
            {"decision_boundary", "dataset"},
            stochastic=True,
            invoker=_dataset_attack_invoker,
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

    def _default_invoke(self, ctx: AttackInvokeCtx) -> torch.Tensor:
        foolbox = ctx.library
        model, inputs, targets, backend = ctx.model, ctx.inputs, ctx.targets, ctx.backend
        kwargs = dict(ctx.call_kwargs)

        attacks_module = foolbox.attacks
        try:
            attack_class = getattr(attacks_module, self.algorithm)
        except AttributeError as error:
            raise ValueError(
                f"{self.algorithm!r} is not a valid foolbox.attacks class name."
            ) from error

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

        # Construction is inside _rethrow so config-required errors (e.g. missing
        # ``steps`` on VirtualAdversarialAttack) are rewritten into actionable messages.
        with self._rethrow():
            attack = attack_class(**self.init_kwargs)
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
