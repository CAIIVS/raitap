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


def test_reference_match_returns_score_of_best_iou_predicted_box() -> None:
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
        iou_threshold=0.5,
    )
    out = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 9.0, 9.0], [50.0, 50.0, 60.0, 60.0]]),
            "scores": torch.tensor([0.8, 0.4]),
            "labels": torch.tensor([1, 1]),
        },
    ]
    value = target(out)
    assert torch.allclose(value, torch.tensor(0.8))


def test_reference_match_zero_when_no_iou_match() -> None:
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
        iou_threshold=0.5,
    )
    out = [
        {
            "boxes": torch.tensor([[50.0, 50.0, 60.0, 60.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        },
    ]
    value = target(out)
    assert torch.allclose(value, torch.tensor(0.0))


def test_reference_match_zero_when_label_mismatch() -> None:
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
        iou_threshold=0.5,
    )
    out = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 9.0, 9.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([2]),
        },
    ]
    value = target(out)
    assert torch.allclose(value, torch.tensor(0.0))


def test_reference_match_picks_highest_iou_when_multiple_matches() -> None:
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
        iou_threshold=0.3,
    )
    out = [
        {
            "boxes": torch.tensor(
                [[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 9.0, 9.0]],
            ),
            "scores": torch.tensor([0.95, 0.6]),
            "labels": torch.tensor([1, 1]),
        },
    ]
    value = target(out)
    # The second box has higher IoU with the reference (~0.81 vs ~0.25), so
    # 0.6 wins despite the lower score.
    assert torch.allclose(value, torch.tensor(0.6))


def test_reference_match_requires_reference_xyxy_and_label() -> None:
    with pytest.raises(ValueError, match="reference_xyxy"):
        DetectionTarget(mode="reference_match")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="reference_label"):
        DetectionTarget(mode="reference_match", reference_xyxy=(0.0, 0.0, 1.0, 1.0))  # type: ignore[call-arg]


def test_reference_match_iou_threshold_default_is_half() -> None:
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
    )
    assert target.iou_threshold == 0.5


def test_reference_match_rejects_invalid_iou_threshold() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        DetectionTarget(
            mode="reference_match",
            reference_xyxy=(0.0, 0.0, 1.0, 1.0),
            reference_label=1,
            iou_threshold=1.5,
        )
    with pytest.raises(ValueError, match="iou_threshold"):
        DetectionTarget(
            mode="reference_match",
            reference_xyxy=(0.0, 0.0, 1.0, 1.0),
            reference_label=1,
            iou_threshold=-0.1,
        )


def test_reference_match_keeps_autograd_graph_when_no_match() -> None:
    """When no predicted box matches the reference, the returned scalar must
    still be connected to the model outputs so gradient explainers (Captum /
    SHAP gradient / Grad-CAM) can backpropagate through scores.
    torch.tensor(0.0) would be a leaf and break the graph."""
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
        iou_threshold=0.5,
    )
    scores = torch.tensor([0.9, 0.4], requires_grad=True)
    out = [
        {
            "boxes": torch.tensor([[50.0, 50.0, 60.0, 60.0], [70.0, 70.0, 80.0, 80.0]]),
            "scores": scores,
            "labels": torch.tensor([1, 1]),
        },
    ]
    value = target(out)
    assert torch.allclose(value, torch.tensor(0.0))
    assert value.requires_grad, "no-match path must keep the graph alive"
    value.backward()
    assert scores.grad is not None
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_reference_match_keeps_autograd_graph_when_boxes_empty() -> None:
    """Empty-predictions branch must also produce a graph-connected zero."""
    target = DetectionTarget(
        mode="reference_match",
        reference_xyxy=(0.0, 0.0, 10.0, 10.0),
        reference_label=1,
        iou_threshold=0.5,
    )
    scores = torch.zeros(0, requires_grad=True)
    out = [
        {
            "boxes": torch.zeros((0, 4)),
            "scores": scores,
            "labels": torch.zeros(0, dtype=torch.int64),
        },
    ]
    value = target(out)
    assert value.requires_grad
