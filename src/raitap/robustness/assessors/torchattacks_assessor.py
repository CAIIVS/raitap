"""Torchattacks adapter for RAITAP robustness assessments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ..contracts import MethodKind, Objective, PerturbationNorm, ThreatModel
from ..exceptions import AssessorBackendIncompatibilityError
from ..semantics import AssessorSemanticsHints
from .base_assessor import EmpiricalAttackAssessor, _prepare_inputs_for_forward

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import nn


class TorchattacksAssessor(EmpiricalAttackAssessor):
    """Single wrapper for ALL torchattacks methods.

    Uses dynamic method loading - no need for class-per-method.
    """

    algorithm_registry: ClassVar[Mapping[str, AssessorSemanticsHints]] = {
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

    def __init__(self, algorithm: str, **init_kwargs: Any) -> None:
        self.algorithm = algorithm
        self.init_kwargs = dict(init_kwargs)

    def check_backend_compat(self, backend: object) -> None:
        if getattr(backend, "supports_torch_autograd", False):
            return
        raise AssessorBackendIncompatibilityError(
            assessor=type(self).__name__,
            backend=type(backend).__name__,
            algorithm=self.algorithm,
            reason=(
                "torchattacks white-box methods require a backend that supports torch autograd. "
                "Use a torch backend (e.g. torch-cpu / torch-cuda / torch-intel) rather than ONNX."
            ),
        )

    def generate_adversarial(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        backend: object | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        try:
            import torchattacks  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise ImportError(
                "TorchattacksAssessor requires the optional dependency 'torchattacks'. "
                "Install it with `uv sync --extra torchattacks` (or `--extra robustness`)."
            ) from error

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
