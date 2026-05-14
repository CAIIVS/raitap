"""Tests for command rendering and mode selection."""

from __future__ import annotations

import pytest

from raitap.deps.command import (
    render_command,
    select_mode,
)


def test_sync_renders_one_extra_flag_per_item() -> None:
    argv, pretty = render_command(mode="sync", extras={"captum", "torch-cpu"})
    assert argv == ["uv", "sync", "--extra", "captum", "--extra", "torch-cpu"]
    assert "captum" in pretty and "torch-cpu" in pretty


def test_add_renders_single_bracketed_call() -> None:
    argv, pretty = render_command(mode="add", extras={"captum", "torch-cpu"})
    assert argv == ["uv", "add", "raitap[captum,torch-cpu]"]
    assert "raitap[captum,torch-cpu]" in pretty


def test_extras_are_sorted_and_deduplicated() -> None:
    argv, _ = render_command(mode="sync", extras={"b", "a"})
    assert argv == ["uv", "sync", "--extra", "a", "--extra", "b"]


def test_empty_extras_still_yields_minimal_command() -> None:
    argv_sync, _ = render_command(mode="sync", extras=set())
    assert argv_sync == ["uv", "sync"]
    argv_add, _ = render_command(mode="add", extras=set())
    assert argv_add == ["uv", "add", "raitap"]


def test_select_mode_auto_uses_is_dev_install(monkeypatch: pytest.MonkeyPatch) -> None:
    from raitap.deps import command as command_module

    monkeypatch.setattr(command_module, "is_dev_install", lambda: True)
    assert select_mode("auto") == "sync"
    monkeypatch.setattr(command_module, "is_dev_install", lambda: False)
    assert select_mode("auto") == "add"


def test_select_mode_explicit() -> None:
    assert select_mode("sync") == "sync"
    assert select_mode("add") == "add"


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError):
        select_mode("nope")  # type: ignore[arg-type]


def test_sync_with_python_version_inserts_p_flag() -> None:
    argv, pretty = render_command(mode="sync", extras={"marabou"}, python_version="3.11")
    assert argv == ["uv", "sync", "-p", "3.11", "--extra", "marabou"]
    assert "-p 3.11" in pretty


def test_add_with_python_version_inserts_p_flag() -> None:
    argv, _ = render_command(mode="add", extras={"marabou"}, python_version="3.11")
    assert argv == ["uv", "add", "-p", "3.11", "raitap[marabou]"]
