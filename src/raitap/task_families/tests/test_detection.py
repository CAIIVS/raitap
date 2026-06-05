from __future__ import annotations

import pytest
import torch

from raitap.task_families.registry import resolve_task_family
from raitap.transparency.contracts import ExplanationOutputSpace
from raitap.types import TaskKind


def test_detection_family_registered() -> None:
    fam = resolve_task_family(TaskKind.detection)
    assert fam.kind is TaskKind.detection
    assert fam.fixed_output_space is ExplanationOutputSpace.DETECTION_BOXES
    assert fam.supports_robustness is False
    assert fam.allows_preprocessing is False
    assert fam.prediction_summaries(payload=object()) is None


def test_validate_payload_requires_list_of_dicts() -> None:
    fam = resolve_task_family(TaskKind.detection)
    fam.validate_payload([{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0)}])
    with pytest.raises(ValueError, match="detection"):
        fam.validate_payload(torch.zeros(2, 3))


def test_validate_labels_rejects_classification_shaped_labels() -> None:
    fam = resolve_task_family(TaskKind.detection)
    # list[dict] / None are detection-shaped → OK.
    fam.validate_labels([{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0)}])
    fam.validate_labels(None)
    # A bare tensor is classification-shaped → model and data disagree.
    with pytest.raises(ValueError, match="classification-shaped"):
        fam.validate_labels(torch.zeros(3))
