"""Friendly error_patterns for config-required foolbox attacks (#266).

Raw errors captured from foolbox 3.3.4:

1. FlexibleDistance attacks (GaussianBlurAttack etc.) without ``distance=``::

       ValueError: unknown distance, please pass `distance` to the attack initializer

   Raised at *call time* inside ``_rethrow``.

2. VirtualAdversarialAttack without ``steps=``::

       TypeError: VirtualAdversarialAttack.__init__() missing 1 required
       positional argument: 'steps'

   Raised at *construction time* (was outside ``_rethrow``); the adapter moves
   construction inside the rethrow block so the pattern fires.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("foolbox")

from raitap.robustness.assessors.foolbox_assessor import FoolboxAssessor
from raitap.utils.errors import AdapterError


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 10)).eval()


def test_flexible_distance_attack_gives_actionable_error() -> None:
    """FlexibleDistance attacks raise a friendly message about constructor.distance."""
    a = FoolboxAssessor("GaussianBlurAttack")
    with pytest.raises(AdapterError, match=r"constructor\.distance"):
        a.generate_adversarial(
            _model(),
            torch.rand(2, 3, 4, 4),
            torch.zeros(2, dtype=torch.long),
            eps=0.3,
        )


def test_virtual_adversarial_attack_gives_actionable_error() -> None:
    """VirtualAdversarialAttack raises a friendly message about constructor.steps."""
    a = FoolboxAssessor("VirtualAdversarialAttack")
    with pytest.raises(AdapterError, match=r"constructor\.steps"):
        a.generate_adversarial(
            _model(),
            torch.rand(2, 3, 4, 4),
            torch.zeros(2, dtype=torch.long),
            eps=0.3,
        )
