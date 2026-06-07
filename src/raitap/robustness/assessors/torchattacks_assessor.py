"""Torchattacks adapter for RAITAP robustness assessments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap import raitap_log
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


def _jsma_invoker(ctx: AttackInvokeCtx) -> torch.Tensor:
    """JSMA hardcodes (labels+1)%10 for untargeted -> wrong unless 10 classes (#266)."""
    model, inputs = ctx.model, ctx.inputs
    n_classes: int | None = None
    try:
        with torch.no_grad():
            out = model(_prepare_inputs_for_forward(inputs[:1], model=model, backend=ctx.backend))
        n_classes = int(out.shape[-1]) if isinstance(out, torch.Tensor) else None
    except Exception:  # undeterminable -> warn + rely on docs caveat
        n_classes = None
    if n_classes is not None and n_classes != 10:
        raise ValueError(
            f"JSMA hardcodes target=(labels+1)%10 for untargeted mode and is only valid on "
            f"10-class models; this model has {n_classes} classes. Use a different attack."
        )
    if n_classes is None:
        raitap_log.warn(
            "JSMA assumes a 10-class model (hardcoded modulo-10 targeting); could not verify "
            "the class count here. Results are invalid if the model is not 10-class."
        )
    return ctx.assessor._default_invoke(ctx)


@robustness_adapter(
    registry_name="torchattacks",
    library="torchattacks",
    algorithm_registry={
        "FGSM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families={"gradient_sign"},
            requires={Capability.AUTOGRAD},
        ),
        "BIM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families={"gradient_sign", "iterative"},
            requires={Capability.AUTOGRAD},
        ),
        "PGD": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families={"gradient_sign", "iterative"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # random_start=True default (uniform init)
        ),
        "PGDL2": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L2,
            families={"gradient_sign", "iterative"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # random_start=True default (uniform init)
        ),
        "MIFGSM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families={"gradient_sign", "iterative", "momentum"},
            requires={Capability.AUTOGRAD},
        ),
        "CW": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L2,
            families={"optimization"},
            requires={Capability.AUTOGRAD},
        ),
        "DeepFool": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L2,
            families={"optimization"},
            requires={Capability.AUTOGRAD},
        ),
        "AutoAttack": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families={"ensemble", "auto"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # default seed=None -> time-seeded; random APGD/FAB/Square init
        ),
        # Square/OnePixel are score-based (conceptually black-box); requires=AUTOGRAD
        # preserves current gating via the torchattacks torch model, could be relaxed later.
        "Square": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.BLACK_BOX_SCORE,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,
            families={"score_based"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # random search
        ),
        "OnePixel": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.BLACK_BOX_SCORE,
            Objective.UNTARGETED,
            PerturbationNorm.L0,
            families={"score_based", "evolutionary"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # differential evolution (unseeded)
        ),
        # --- #266: expanded coverage (26 added). Hints verified against
        # torchattacks 3.5.1 installed source (signatures + RNG/norm bodies). ---
        "APGD": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # norm='Linf' default kwarg
            families={"gradient_sign", "iterative", "auto"},
            requires={Capability.AUTOGRAD},
            # Draws random init but self-pins seed=0 by default -> reproducible
            # (cf. AutoAttack seed=None -> stochastic). Determinism-verified.
        ),
        "APGDT": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.TARGETED,  # APGDT is the targeted APGD variant
            PerturbationNorm.LINF,  # norm='Linf' default kwarg
            families={"gradient_sign", "iterative", "auto"},
            requires={Capability.AUTOGRAD},
            # seed=0 default -> reproducible (determinism-verified).
        ),
        "DIFGSM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step, transfer attack
            families={"gradient_sign", "iterative", "momentum", "transfer"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # input_diversity uses torch.rand/randint every step
        ),
        "EADEN": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L1,  # elastic-net (L1+L2), L1-dominant sparsity
            families={"optimization"},
            requires={Capability.AUTOGRAD},
            # deterministic optimisation (no default RNG in source).
        ),
        "EADL1": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L1,  # L1 elastic-net variant
            families={"optimization"},
            requires={Capability.AUTOGRAD},
            # deterministic optimisation (no default RNG in source).
        ),
        "EOTPGD": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step
            families={"gradient_sign", "iterative"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # random_start=True default (uniform_ init)
        ),
        "FAB": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # norm='Linf' default kwarg
            families={"optimization", "auto"},
            requires={Capability.AUTOGRAD},
            # seed=0 default -> reproducible (determinism-verified).
        ),
        "FFGSM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded single-step sign
            families={"gradient_sign"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # randn_like().uniform_ random init, unseeded
        ),
        "GN": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L2,  # additive Gaussian noise; no clean norm, L2 closest
            families={"noise"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # std * torch.randn_like(images) every call
        ),
        "JSMA": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L0,  # saliency-map few-pixel perturbation
            families={"saliency"},
            requires={Capability.AUTOGRAD},
            invoker=_jsma_invoker,  # guard: hardcoded (labels+1)%10 -> 10-class only
            # deterministic saliency map (no default RNG).
        ),
        "Jitter": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step
            families={"gradient_sign", "iterative"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # random_start=True default + randn jitter noise
        ),
        "NIFGSM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step, transfer attack
            families={"gradient_sign", "iterative", "momentum", "transfer"},
            requires={Capability.AUTOGRAD},
            # Nesterov momentum, deterministic (no default RNG).
        ),
        "PGDRS": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step (smoothed)
            families={"gradient_sign", "iterative", "randomized_smoothing"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # samples gaussian noise batch each step
        ),
        "PGDRSL2": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L2,  # L2 randomized-smoothing PGD
            families={"gradient_sign", "iterative", "randomized_smoothing"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # samples gaussian noise batch each step
        ),
        "PIFGSM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # max_epsilon Linf-bounded transfer attack
            families={"gradient_sign", "iterative", "transfer"},
            requires={Capability.AUTOGRAD},
            # patch-wise iterative, deterministic (no default RNG).
        ),
        "PIFGSMPP": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # max_epsilon Linf-bounded transfer attack
            families={"gradient_sign", "iterative", "transfer"},
            requires={Capability.AUTOGRAD},
            # patch-wise++ iterative, deterministic (no default RNG).
        ),
        "Pixle": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.BLACK_BOX_SCORE,  # random search, no backprop (verified)
            Objective.UNTARGETED,
            PerturbationNorm.L0,  # rearranges a few pixels
            families={"score_based"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # np.random pixel search, unseeded
        ),
        "RFGSM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step
            families={"gradient_sign", "iterative"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # randn_like().sign() random init step every call
        ),
        "SINIFGSM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step, transfer attack
            families={"gradient_sign", "iterative", "momentum", "transfer"},
            requires={Capability.AUTOGRAD},
            # scale-invariant Nesterov, deterministic (no default RNG).
        ),
        "SPSA": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.BLACK_BOX_SCORE,  # finite-difference, no backprop (verified)
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps Linf-bounded
            families={"score_based"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # Bernoulli perturbation directions sampled each iter
        ),
        "SparseFool": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.L0,  # sparse few-pixel perturbation
            families={"optimization", "sparse"},
            requires={Capability.AUTOGRAD},
            # deterministic geometric solver (no default RNG).
        ),
        "TIFGSM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step, transfer attack
            families={"gradient_sign", "iterative", "momentum", "transfer"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # input_diversity uses torch.rand/randint every step
        ),
        "TPGD": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step (KL/TRADES)
            families={"gradient_sign", "iterative"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # 0.001 * randn_like random init every call (unseeded)
        ),
        "UPGD": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step
            families={"gradient_sign", "iterative"},
            requires={Capability.AUTOGRAD},
            # random_start=False default -> deterministic (no default RNG).
        ),
        "VMIFGSM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step, transfer attack
            families={"gradient_sign", "iterative", "momentum", "variance_tuned"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # variance neighbour sampling (uniform_) every step
        ),
        "VNIFGSM": AssessorSemanticsHints(
            AssessmentKind.EMPIRICAL_ATTACK,
            ThreatModel.WHITE_BOX,
            Objective.UNTARGETED,
            PerturbationNorm.LINF,  # eps-bounded sign step, transfer attack
            families={"gradient_sign", "iterative", "momentum", "variance_tuned"},
            requires={Capability.AUTOGRAD},
            stochastic=True,  # variance neighbour sampling (uniform_) every step
        ),
    },
)
class TorchattacksAssessor(EmpiricalAttackAssessor):
    """Single wrapper for ALL torchattacks methods.

    Uses dynamic method loading - no need for class-per-method.
    """

    def __init__(self, algorithm: str, **init_kwargs: Any) -> None:
        self.algorithm = algorithm
        self.init_kwargs = dict(init_kwargs)

    def _default_invoke(self, ctx: AttackInvokeCtx) -> torch.Tensor:
        torchattacks = ctx.library
        model, inputs, targets, backend = ctx.model, ctx.inputs, ctx.targets, ctx.backend
        kwargs = dict(ctx.call_kwargs)

        try:
            attack_class = getattr(torchattacks, self.algorithm)
        except AttributeError as error:
            raise ValueError(
                f"{self.algorithm!r} is not a valid torchattacks attack name."
            ) from error

        attack = attack_class(model, **self.init_kwargs)

        # Per-call kwargs that need special handling before calling the attack object.
        normalization = kwargs.pop("normalization", None)
        if normalization:
            attack.set_normalization_used(**normalization)

        target_kwargs = self._maybe_set_targeted(attack, kwargs)

        # torchattacks methods (PGDL2, CW, DeepFool, Square, ...) call ``.view(...)``
        # internally, which needs contiguous memory. RAITAP's image loader produces
        # NCHW tensors via HWC->CHW transpose, so we make inputs contiguous defensively
        # after routing them through the backend's ``_prepare_inputs`` for device placement.
        inputs_dev = _prepare_inputs_for_forward(inputs, model=model, backend=backend).contiguous()
        targets_dev = _prepare_inputs_for_forward(
            targets, model=model, backend=backend
        ).contiguous()

        with self._rethrow():
            if target_kwargs is not None:
                adversarial = attack(
                    inputs_dev,
                    _prepare_inputs_for_forward(
                        target_kwargs, model=model, backend=backend
                    ).contiguous(),
                )
            else:
                adversarial = attack(inputs_dev, targets_dev)
        return adversarial.detach()

    def _maybe_set_targeted(
        self,
        attack: Any,
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        target_labels = kwargs.pop("target_labels", None)
        target_classes = kwargs.pop("target_classes", None)
        if target_labels is None and target_classes is None:
            return None
        # torchattacks' targeted-mode plumbing varies per attack; the canonical entry
        # point is `set_mode_targeted_by_label` if present, otherwise we just pass
        # the target tensor through to attack().
        if hasattr(attack, "set_mode_targeted_by_label"):
            attack.set_mode_targeted_by_label(quiet=True)
        if target_labels is not None:
            return _to_long_tensor(target_labels)
        return _to_long_tensor(target_classes)


def _to_long_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(torch.long)
    if isinstance(value, int):
        return torch.tensor([value], dtype=torch.long)
    return torch.tensor(list(value), dtype=torch.long)
