from __future__ import annotations

import pytest
import torch

from raitap.pipeline.outputs import OutputKind
from raitap.task_families.classification import ClassificationFamily
from raitap.task_families.registry import resolve_task_family
from raitap.types import TaskKind


def test_classification_family_registered() -> None:
    fam = resolve_task_family(TaskKind.classification)
    assert fam.kind is TaskKind.classification
    assert fam.fixed_output_space is None
    assert fam.supports_robustness is True
    assert fam.allows_preprocessing is True


def test_validate_payload_requires_tensor() -> None:
    fam = resolve_task_family(TaskKind.classification)
    fam.validate_payload(torch.zeros(2, 3))  # ok
    with pytest.raises(ValueError, match="classification"):
        fam.validate_payload(None)


def test_validate_labels_rejects_detection_shaped_labels() -> None:
    fam = resolve_task_family(TaskKind.classification)
    # Tensor / None are classification-shaped → OK.
    fam.validate_labels(torch.zeros(3))
    fam.validate_labels(None)
    # A list[dict] is detection-shaped → model and data disagree.
    with pytest.raises(ValueError, match="detection-shaped"):
        fam.validate_labels([{"boxes": torch.zeros(0, 4)}])


def test_probabilities_skip_softmax_in_prediction_summaries() -> None:
    family = ClassificationFamily()
    # A peaked distribution: class 1 prob 0.9. Softmax of this would shrink it.
    probs = torch.tensor([[0.1, 0.9]])
    rows = family.prediction_summaries(
        probs, sample_ids=None, targets=None, output_kind=OutputKind.PROBABILITIES
    )
    assert rows is not None
    assert rows[0].predicted_class == 1
    assert abs(rows[0].confidence - 0.9) < 1e-6


def test_logits_apply_softmax_in_prediction_summaries() -> None:
    family = ClassificationFamily()
    logits = torch.tensor([[0.1, 0.9]])
    rows = family.prediction_summaries(
        logits, sample_ids=None, targets=None, output_kind=OutputKind.LOGITS
    )
    assert rows is not None
    expected = float(torch.softmax(logits, dim=1).max())
    assert abs(rows[0].confidence - expected) < 1e-6


def test_omitted_output_kind_defaults_to_softmax() -> None:
    # Direct callers that omit output_kind get the historical logits behaviour.
    family = ClassificationFamily()
    logits = torch.tensor([[0.1, 0.9]])
    rows = family.prediction_summaries(logits, sample_ids=None, targets=None)
    assert rows is not None
    expected = float(torch.softmax(logits, dim=1).max())
    assert abs(rows[0].confidence - expected) < 1e-6
