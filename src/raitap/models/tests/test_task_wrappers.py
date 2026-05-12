"""Tests for DetectionTarget — scalar reducer for detection model outputs."""

from __future__ import annotations

import pytest
import torch

from raitap.models.task_wrappers import DetectionTarget


def _sample_detection_output() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 12.0, 12.0]]),
            "scores": torch.tensor([0.9, 0.4]),
            "labels": torch.tensor([1, 2]),
        },
    ]


def test_class_score_returns_score_at_box_index() -> None:
    target = DetectionTarget(box_idx=0, mode="class_score")
    value = target(_sample_detection_output())
    assert torch.allclose(value, torch.tensor(0.9))


def test_class_score_returns_zero_for_missing_box() -> None:
    target = DetectionTarget(box_idx=99, mode="class_score")
    value = target(_sample_detection_output())
    assert torch.allclose(value, torch.tensor(0.0))


def test_objectness_sums_scores_in_batch() -> None:
    target = DetectionTarget(box_idx=0, mode="objectness")
    value = target(_sample_detection_output())
    assert torch.allclose(value, torch.tensor(0.9 + 0.4))


def test_bbox_l2_returns_squared_norm_of_first_box() -> None:
    target = DetectionTarget(box_idx=0, mode="bbox_l2")
    value = target(_sample_detection_output())
    expected = torch.tensor(0.0**2 + 0.0**2 + 10.0**2 + 10.0**2)
    assert torch.allclose(value, expected)


def test_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        DetectionTarget(box_idx=0, mode="nonsense")  # type: ignore[arg-type]


def test_rejects_non_list_output() -> None:
    target = DetectionTarget(box_idx=0, mode="class_score")
    with pytest.raises(TypeError):
        target(torch.zeros(3))  # type: ignore[arg-type]


def test_objectness_with_empty_scores_returns_zero() -> None:
    target = DetectionTarget(box_idx=0, mode="objectness")
    value = target(
        [
            {
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.int64),
            }
        ]
    )
    assert torch.allclose(value, torch.tensor(0.0))


def test_objectness_sums_across_multi_sample_batch() -> None:
    target = DetectionTarget(box_idx=0, mode="objectness")
    sample_a = {
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "scores": torch.tensor([0.5]),
        "labels": torch.tensor([1]),
    }
    sample_b = {
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]),
        "scores": torch.tensor([0.3, 0.2]),
        "labels": torch.tensor([1, 2]),
    }
    value = target([sample_a, sample_b])
    assert torch.allclose(value, torch.tensor(0.5 + 0.3 + 0.2))


def test_empty_batch_returns_zero() -> None:
    target = DetectionTarget(box_idx=0, mode="class_score")
    value = target([])
    assert torch.allclose(value, torch.tensor(0.0))
