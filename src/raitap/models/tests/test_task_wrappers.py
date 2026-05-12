"""Tests for DetectionTarget — scalar reducer for detection model outputs."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from raitap.models.task_wrappers import DetectionTarget, ScalarDetectionWrapper


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


class _FakeDetector(nn.Module):
    def forward(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        batch_size = images.shape[0]
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": torch.tensor([0.7]),
                "labels": torch.tensor([5]),
            }
            for _ in range(batch_size)
        ]


def test_scalar_detection_wrapper_returns_batched_logit_tensor() -> None:
    wrapper = ScalarDetectionWrapper(
        _FakeDetector(), target=DetectionTarget(box_idx=0, mode="class_score")
    )
    images = torch.zeros(2, 3, 8, 8)
    out = wrapper(images)
    # Shape contract: (batch, 1) so existing classification-shaped explainers
    # (which call ``output[:, target_class]``) work unchanged.
    assert out.shape == (2, 1)
    assert torch.allclose(out, torch.tensor([[0.7], [0.7]]))


def test_scalar_detection_wrapper_is_an_nn_module() -> None:
    wrapper = ScalarDetectionWrapper(
        _FakeDetector(), target=DetectionTarget(box_idx=0, mode="objectness")
    )
    assert isinstance(wrapper, nn.Module)


def test_scalar_detection_wrapper_eval_propagates_to_inner_model() -> None:
    detector = _FakeDetector()
    detector.train()
    wrapper = ScalarDetectionWrapper(
        detector, target=DetectionTarget(box_idx=0, mode="class_score")
    )
    wrapper.eval()
    assert not detector.training
