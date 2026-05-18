"""Tests for the typed ForwardOutput dataclass."""

from __future__ import annotations

import pytest
import torch

from raitap.pipeline.outputs import ForwardOutput
from raitap.types import TaskKind


def test_classification_forward_output_requires_predictions_tensor() -> None:
    with pytest.raises(ValueError, match="predictions_tensor"):
        ForwardOutput(task_kind=TaskKind.classification, batch_size=4)


def test_detection_forward_output_requires_detection_predictions() -> None:
    with pytest.raises(ValueError, match="detection_predictions"):
        ForwardOutput(task_kind=TaskKind.detection, batch_size=2)


def test_classification_forward_output_round_trip() -> None:
    predictions = torch.zeros(4, 10)
    out = ForwardOutput(
        task_kind=TaskKind.classification,
        batch_size=4,
        predictions_tensor=predictions,
    )
    assert out.task_kind is TaskKind.classification
    assert out.batch_size == 4
    assert out.predictions_tensor is not None
    assert torch.equal(out.predictions_tensor, predictions)
    assert out.detection_predictions is None


def test_detection_forward_output_round_trip() -> None:
    detection_predictions = [
        {
            "boxes": torch.zeros((1, 4)),
            "scores": torch.zeros(1),
            "labels": torch.zeros(1, dtype=torch.int64),
        },
        {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.int64),
        },
    ]
    out = ForwardOutput(
        task_kind=TaskKind.detection,
        batch_size=2,
        detection_predictions=detection_predictions,
    )
    assert out.task_kind is TaskKind.detection
    assert out.batch_size == 2
    assert out.detection_predictions == detection_predictions
    assert out.predictions_tensor is None
