from __future__ import annotations

import pytest
import torch

import raitap.task_families.detection  # noqa: F401 - register the family
from raitap.task_families.registry import resolve_task_family
from raitap.transparency.contracts import ExplanationOutputSpace
from raitap.types import TaskKind


def test_detection_family_registered() -> None:
    fam = resolve_task_family(TaskKind.detection)
    assert fam.kind is TaskKind.detection
    assert fam.output_space is ExplanationOutputSpace.DETECTION_BOXES
    assert fam.supports_robustness() is False
    assert fam.allows_preprocessing is False
    assert fam.prediction_summaries(payload=object()) is None


def test_validate_payload_requires_list_of_dicts() -> None:
    fam = resolve_task_family(TaskKind.detection)
    fam.validate_payload([{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0)}])
    with pytest.raises(ValueError, match="detection"):
        fam.validate_payload(torch.zeros(2, 3))
