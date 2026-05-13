"""Tests for python_version.pick_python_version."""

from __future__ import annotations

import sys
import textwrap
from typing import TYPE_CHECKING

from raitap.configs.extras import python_version as pv

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "pyproject.toml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_no_extras_no_pin(tmp_path: Path) -> None:
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
    assert pv.pick_python_version(py, ["captum"]) is None


def test_marker_caps_python_on_compatible_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the linux branch so the marker is not platform-skipped.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(pv, "_base_env", lambda: {**_linux_env(), "sys_platform": "linux"})
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
    assert pv.pick_python_version(py, ["marabou"]) == "3.11"


def test_marker_platform_skipped_no_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # On win32 the marabou marker is always False → no Python constraint.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(pv, "_base_env", lambda: {**_linux_env(), "sys_platform": "win32"})
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
    assert pv.pick_python_version(py, ["marabou"]) is None


def test_unselected_extras_dont_constrain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(pv, "_base_env", lambda: {**_linux_env(), "sys_platform": "linux"})
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
        captum = ["captum>=0.7.0"]
    """,
    )
    assert pv.pick_python_version(py, ["captum"]) is None


def test_missing_requires_python_uses_default_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(pv, "_base_env", lambda: {**_linux_env(), "sys_platform": "linux"})
    py = _write(
        tmp_path,
        """
        [project]
        name = "x"
        [project.optional-dependencies]
        marabou = [
            "maraboupy>=2.0.0,<3.0; python_full_version < '3.12'",
        ]
    """,
    )
    assert pv.pick_python_version(py, ["marabou"]) == "3.11"


def test_empty_extras_no_pin(tmp_path: Path) -> None:
    py = _write(
        tmp_path,
        """
        [project]
        name = "x"
        requires-python = ">=3.11,<3.14"
        [project.optional-dependencies]
    """,
    )
    assert pv.pick_python_version(py, []) is None


def _linux_env() -> dict[str, str]:
    return {
        "implementation_name": "cpython",
        "implementation_version": "3.13.0",
        "os_name": "linux",
        "platform_machine": "x86_64",
        "platform_release": "6.0",
        "platform_system": "Linux",
        "platform_version": "#1",
        "platform_python_implementation": "CPython",
    }
