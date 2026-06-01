from __future__ import annotations

import pytest
import torch

from raitap.transparency.contracts import DetectionBox
from raitap.transparency.detection_labels import (
    enrich_detection_box,
    label_name_for,
    match_box_to_gt,
    resolve_category_names,
)


def test_resolve_prefers_explicit_over_backend() -> None:
    assert resolve_category_names(["a", "b"], ["x", "y"]) == ["a", "b"]


def test_resolve_falls_back_to_backend_when_no_explicit() -> None:
    assert resolve_category_names(None, ["x", "y"]) == ["x", "y"]


def test_resolve_returns_none_when_neither_present() -> None:
    assert resolve_category_names(None, None) is None


def test_resolve_copies_to_list() -> None:
    src = ("a", "b")
    out = resolve_category_names(src, None)
    assert out == ["a", "b"]
    assert isinstance(out, list)


def test_label_name_for_in_range() -> None:
    assert label_name_for(38, ["__background__"] + [f"c{i}" for i in range(1, 91)]) == "c38"


def test_label_name_for_none_map_returns_none() -> None:
    assert label_name_for(3, None) is None


def test_label_name_for_out_of_range_returns_none() -> None:
    assert label_name_for(99, ["a", "b"]) is None
    assert label_name_for(-1, ["a", "b"]) is None


def _raw_box(label_index: int = 38) -> DetectionBox:
    return DetectionBox(
        display_index=0,
        raw_index=2,
        xyxy=(0.0, 0.0, 1.0, 1.0),
        score=0.99,
        label_index=label_index,
        label_name=None,
    )


def test_enrich_sets_label_name_from_names() -> None:
    names = ["__background__"] + [f"c{i}" for i in range(1, 91)]
    out = enrich_detection_box(_raw_box(38), category_names=names)
    assert out.label_name == "c38"


def test_enrich_label_name_none_without_names() -> None:
    out = enrich_detection_box(_raw_box(38), category_names=None)
    assert out.label_name is None


def test_enrich_returns_new_frozen_instance() -> None:
    raw = _raw_box(38)
    out = enrich_detection_box(raw, category_names=["__background__", "kite"])
    assert out is not raw
    assert out.label_index == raw.label_index  # untouched fields preserved


def test_match_returns_best_iou_gt_class_agnostic() -> None:
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    gt_labels = torch.tensor([7, 3])
    out = match_box_to_gt((0.0, 0.0, 10.0, 10.0), gt_boxes, gt_labels, iou_threshold=0.5)
    assert out is not None
    idx, iou = out
    assert idx == 7
    assert iou == pytest.approx(1.0)


def test_match_below_threshold_returns_none() -> None:
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_labels = torch.tensor([7])
    out = match_box_to_gt((100.0, 100.0, 110.0, 110.0), gt_boxes, gt_labels, iou_threshold=0.5)
    assert out is None


def test_match_empty_gt_returns_none() -> None:
    out = match_box_to_gt(
        (0.0, 0.0, 10.0, 10.0),
        torch.zeros((0, 4)),
        torch.zeros(0, dtype=torch.int64),
        iou_threshold=0.5,
    )
    assert out is None


def test_match_picks_highest_iou_among_several() -> None:
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 9.0, 9.0]])
    gt_labels = torch.tensor([1, 2])
    out = match_box_to_gt((0.0, 0.0, 9.0, 9.0), gt_boxes, gt_labels, iou_threshold=0.1)
    assert out is not None
    assert out[0] == 2  # exact overlap with the second box


NAMES = ["__background__"] + [f"c{i}" for i in range(1, 91)]


def test_enrich_matches_gt_class_agnostic() -> None:
    gt = {"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "labels": torch.tensor([20])}
    out = enrich_detection_box(
        _raw_box(38), category_names=NAMES, gt_for_sample=gt, iou_threshold=0.5
    )
    assert out.gt_evaluated is True
    assert out.true_label_index == 20
    assert out.true_label_name == "c20"  # GT class differs from pred -> disagreement shown
    assert out.true_match_iou == pytest.approx(1.0)


def test_enrich_no_match_is_false_positive() -> None:
    gt = {"boxes": torch.tensor([[100.0, 100.0, 101.0, 101.0]]), "labels": torch.tensor([5])}
    out = enrich_detection_box(
        _raw_box(38), category_names=NAMES, gt_for_sample=gt, iou_threshold=0.5
    )
    assert out.gt_evaluated is True  # GT present for the sample
    assert out.true_label_index is None  # nothing overlapped -> no match
    assert out.true_label_name is None
    assert out.true_match_iou is None


def test_enrich_no_gt_leaves_evaluated_false() -> None:
    out = enrich_detection_box(_raw_box(38), category_names=NAMES, gt_for_sample=None)
    assert out.gt_evaluated is False
    assert out.true_label_index is None
