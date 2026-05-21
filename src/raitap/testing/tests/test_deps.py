from __future__ import annotations

from raitap.testing import requires


def test_requires_present_dep_does_not_skip() -> None:
    mark = requires("pytest").mark
    assert mark.name == "skipif"
    assert mark.args == (False,)  # present -> condition False -> not skipped


def test_requires_missing_dep_marks_skip() -> None:
    mark = requires("definitely_not_a_real_module_xyz").mark
    assert mark.name == "skipif"
    assert mark.args == (True,)  # missing -> condition True -> skipped
    assert "definitely_not_a_real_module_xyz" in mark.kwargs["reason"]
