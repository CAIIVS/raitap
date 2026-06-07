"""Empirical attack dispatch routes to a per-entry invoker or the default (#266)."""

from __future__ import annotations

import pytest
import torch

from raitap.robustness.assessors.base_assessor import AttackInvokeCtx
from raitap.robustness.assessors.torchattacks_assessor import TorchattacksAssessor


def test_custom_invoker_overrides_default() -> None:
    sentinel = torch.zeros(1, 3, 4, 4)

    def fake_invoker(ctx: AttackInvokeCtx) -> torch.Tensor:
        assert isinstance(ctx, AttackInvokeCtx)
        assert ctx.assessor.algorithm == "FGSM"  # pyright: ignore[reportAttributeAccessIssue]  # ctx carries the assessor
        return sentinel

    a = TorchattacksAssessor("FGSM")
    hints = type(a).algorithm_registry["FGSM"]
    original = hints.invoker
    object.__setattr__(hints, "invoker", fake_invoker)  # frozen dataclass
    try:
        out = a.generate_adversarial(
            torch.nn.Conv2d(3, 3, 1), torch.zeros(1, 3, 4, 4), torch.zeros(1, dtype=torch.long)
        )
        assert out is sentinel
    finally:
        object.__setattr__(hints, "invoker", original)


def test_default_path_runs_when_no_invoker() -> None:
    pytest.importorskip("torchattacks")
    a = TorchattacksAssessor("FGSM")  # FGSM has no custom invoker
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 10)).eval()
    out = a.generate_adversarial(model, torch.rand(2, 3, 4, 4), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 3, 4, 4)
