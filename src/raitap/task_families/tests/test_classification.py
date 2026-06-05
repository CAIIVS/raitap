from __future__ import annotations

import pytest
import torch

from raitap.task_families.registry import resolve_task_family
from raitap.types import TaskKind


def test_classification_family_registered() -> None:
    fam = resolve_task_family(TaskKind.classification)
    assert fam.kind is TaskKind.classification
    assert fam.fixed_output_space is None
    assert fam.supports_robustness() is True
    assert fam.allows_preprocessing is True


def test_validate_payload_requires_tensor() -> None:
    fam = resolve_task_family(TaskKind.classification)
    fam.validate_payload(torch.zeros(2, 3))  # ok
    with pytest.raises(ValueError, match="classification"):
        fam.validate_payload(None)
