"""Tests for availability.check_platform_availability."""

from __future__ import annotations

import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from raitap.configs.extras import availability
from raitap.configs.extras import python_version as pv

if TYPE_CHECKING:
    from pathlib import Path


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "pyproject.toml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _force(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(
        pv,
        "_base_env",
        lambda: {
            "implementation_name": "cpython",
            "implementation_version": "3.13.0",
            "os_name": platform,
            "platform_machine": "x86_64",
            "platform_release": "1",
            "platform_system": platform,
            "platform_version": "1",
            "platform_python_implementation": "CPython",
            "sys_platform": platform,
        },
    )


def test_unmarked_requirement_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force(monkeypatch, "linux")
    py = _write(
        tmp_path,
        """
        [project]
        name = "x"
        requires-python = ">=3.11,<3.14"
        [project.optional-dependencies]
        captum = ["captum>=0.7.0"]
        """,
    )
    availability.check_platform_availability(py, ["captum"])


def test_marker_passing_on_one_python_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force(monkeypatch, "linux")
    py = _write(
        tmp_path,
        """
        [project]
        name = "x"
        requires-python = ">=3.11,<3.14"
        [project.optional-dependencies]
        marabou = [
            "maraboupy>=2.0.0,<3.0; python_full_version < '3.12' and sys_platform != 'win32'",
        ]
        """,
    )
    availability.check_platform_availability(py, ["marabou"])


def test_marker_false_everywhere_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force(monkeypatch, "win32")
    py = _write(
        tmp_path,
        """
        [project]
        name = "x"
        requires-python = ">=3.11,<3.14"
        [project.optional-dependencies]
        marabou = [
            "maraboupy>=2.0.0,<3.0; python_full_version < '3.12' and sys_platform != 'win32'",
            "onnx>=1.15",
        ]
        """,
    )
    with pytest.raises(availability.ExtraUnavailableError) as excinfo:
        availability.check_platform_availability(py, ["marabou"])
    msg = str(excinfo.value)
    assert "marabou" in msg
    assert "maraboupy" in msg
    assert "win32" in msg


def test_unselected_extras_not_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force(monkeypatch, "win32")
    py = _write(
        tmp_path,
        """
        [project]
        name = "x"
        requires-python = ">=3.11,<3.14"
        [project.optional-dependencies]
        marabou = [
            "maraboupy>=2.0.0,<3.0; sys_platform != 'win32'",
        ]
        captum = ["captum>=0.7.0"]
        """,
    )
    availability.check_platform_availability(py, ["captum"])
