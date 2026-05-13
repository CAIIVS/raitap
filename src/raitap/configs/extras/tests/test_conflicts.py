"""Tests for extras conflict validation against [tool.uv].conflicts."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from raitap.configs.extras.conflicts import (
    ExtrasConflictError,
    load_conflict_groups,
    validate_conflicts,
)


def _write(tmp_path: Path, body: str) -> Path:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(textwrap.dedent(body), encoding="utf-8")
    return pyproject


def test_load_conflict_groups(tmp_path: Path) -> None:
    py = _write(tmp_path, """
        [tool.uv]
        conflicts = [
            [{ extra = "torch-cpu" }, { extra = "torch-cuda" }],
            [{ extra = "onnx-cpu" }, { extra = "torch-cuda" }],
        ]
    """)
    groups = load_conflict_groups(py)
    assert groups == [
        frozenset({"torch-cpu", "torch-cuda"}),
        frozenset({"onnx-cpu", "torch-cuda"}),
    ]


def test_load_conflict_groups_missing_table_returns_empty(tmp_path: Path) -> None:
    py = _write(tmp_path, "[project]\nname = \"x\"\n")
    assert load_conflict_groups(py) == []


def test_validate_passes_with_single_member(tmp_path: Path) -> None:
    py = _write(tmp_path, """
        [tool.uv]
        conflicts = [[{ extra = "torch-cpu" }, { extra = "torch-cuda" }]]
    """)
    validate_conflicts({"torch-cpu", "captum"}, py, origins={})


def test_validate_raises_on_two_torch_backends(tmp_path: Path) -> None:
    py = _write(tmp_path, """
        [tool.uv]
        conflicts = [[
            { extra = "torch-cpu" },
            { extra = "torch-cuda" },
            { extra = "torch-intel" },
        ]]
    """)
    with pytest.raises(ExtrasConflictError) as excinfo:
        validate_conflicts(
            {"torch-cpu", "torch-cuda"},
            py,
            origins={"torch-cpu": "model.source ext", "torch-cuda": "--hardware cuda"},
        )
    msg = str(excinfo.value)
    assert "torch-cpu" in msg and "torch-cuda" in msg
    assert "model.source ext" in msg
    assert "--hardware cuda" in msg


def test_validate_cross_family(tmp_path: Path) -> None:
    py = _write(tmp_path, """
        [tool.uv]
        conflicts = [[{ extra = "onnx-cpu" }, { extra = "torch-cuda" }]]
    """)
    with pytest.raises(ExtrasConflictError):
        validate_conflicts({"onnx-cpu", "torch-cuda"}, py, origins={})
