"""Tests for the typed ForwardOutput dataclass."""

from __future__ import annotations

import pytest
import torch

from raitap.pipeline.outputs import ForwardOutput
from raitap.types import TaskKind


def test_generic_payload_classification_roundtrip() -> None:
    logits = torch.zeros(4, 10)
    out = ForwardOutput(task_kind=TaskKind.classification, batch_size=4, payload=logits)
    assert out.payload is logits
    assert out.as_classification() is logits


def test_generic_payload_validates_on_construction() -> None:
    with pytest.raises(ValueError, match="classification"):
        ForwardOutput(task_kind=TaskKind.classification, batch_size=1, payload=None)


def test_detection_payload_accessor() -> None:
    preds = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0)}]
    out = ForwardOutput(task_kind=TaskKind.detection, batch_size=1, payload=preds)
    assert out.as_detection() == preds
