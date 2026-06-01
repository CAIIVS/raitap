from __future__ import annotations

from raitap.transparency.contracts import DetectionBox
from raitap.transparency.detection_labels import (
    enrich_detection_box,
    label_name_for,
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
