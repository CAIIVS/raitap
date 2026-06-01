from __future__ import annotations

from raitap.transparency.detection_labels import label_name_for, resolve_category_names


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
