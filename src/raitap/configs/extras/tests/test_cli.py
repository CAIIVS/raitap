"""Tests for the raitap-deps CLI orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from raitap.configs.extras import __main__ as cli


@pytest.fixture
def fake_compose(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "model": {"source": "model.pt"},
        "transparency": {"ig": {"_target_": "CaptumExplainer", "algorithm": "IG"}},
        "metrics": {"_target_": "ClassificationMetrics", "task": "multiclass"},
        "reporting": {"_target_": "HTMLReporter", "filename": "r"},
    }
    monkeypatch.setattr(cli, "_compose_config", lambda **_k: cfg)
    return cfg


def test_dry_run_does_not_exec(
    monkeypatch: pytest.MonkeyPatch, fake_compose: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "detect_hardware", lambda: "cpu")
    monkeypatch.setattr(cli, "is_dev_install", lambda: True)
    run_mock = MagicMock()
    monkeypatch.setattr(cli.subprocess, "run", run_mock)
    rc = cli.main(["--dry-run", "--hardware", "cpu"])
    assert rc == 0
    run_mock.assert_not_called()
    out = capsys.readouterr().out
    assert "uv sync" in out
    assert "captum" in out
    assert "metrics" in out
    assert "jinja" in out
    assert "torch-cpu" in out


def test_default_executes(
    monkeypatch: pytest.MonkeyPatch, fake_compose: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "detect_hardware", lambda: "cpu")
    monkeypatch.setattr(cli, "is_dev_install", lambda: True)
    completed = MagicMock(returncode=0)
    run_mock = MagicMock(return_value=completed)
    monkeypatch.setattr(cli.subprocess, "run", run_mock)
    rc = cli.main(["--hardware", "cpu"])
    assert rc == 0
    run_mock.assert_called_once()
    argv = run_mock.call_args.args[0]
    assert argv[:2] == ["uv", "sync"]


def test_hardware_override(
    monkeypatch: pytest.MonkeyPatch, fake_compose: dict[str, Any]
) -> None:
    probe_mock = MagicMock(return_value="cuda")
    monkeypatch.setattr(cli, "detect_hardware", probe_mock)
    monkeypatch.setattr(cli, "is_dev_install", lambda: True)
    monkeypatch.setattr(cli.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    cli.main(["--hardware", "cuda", "--dry-run"])
    probe_mock.assert_not_called()


def test_auto_hardware_probes(
    monkeypatch: pytest.MonkeyPatch, fake_compose: dict[str, Any]
) -> None:
    probe_mock = MagicMock(return_value="cuda")
    monkeypatch.setattr(cli, "detect_hardware", probe_mock)
    monkeypatch.setattr(cli, "is_dev_install", lambda: True)
    monkeypatch.setattr(cli.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    cli.main(["--dry-run"])
    probe_mock.assert_called_once()


def test_mode_add(
    monkeypatch: pytest.MonkeyPatch, fake_compose: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "detect_hardware", lambda: "cpu")
    monkeypatch.setattr(cli, "is_dev_install", lambda: True)
    monkeypatch.setattr(cli.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    rc = cli.main(["--mode", "add", "--hardware", "cpu", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "uv add" in out
    assert "raitap[" in out


def test_conflict_aborts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = {
        "model": {"source": "x.pt"},
    }
    monkeypatch.setattr(cli, "_compose_config", lambda **_k: cfg)
    monkeypatch.setattr(cli, "detect_hardware", lambda: "cpu")
    monkeypatch.setattr(cli, "is_dev_install", lambda: True)

    def fake_infer(_cfg: Any, **_k: Any) -> tuple[set[str], dict[str, str]]:
        return {"torch-cpu", "torch-cuda"}, {
            "torch-cpu": "test",
            "torch-cuda": "test",
        }

    monkeypatch.setattr(cli, "infer_extras", fake_infer)
    rc = cli.main(["--hardware", "cpu", "--dry-run"])
    assert rc != 0


def test_exec_propagates_returncode(
    monkeypatch: pytest.MonkeyPatch, fake_compose: dict[str, Any]
) -> None:
    monkeypatch.setattr(cli, "detect_hardware", lambda: "cpu")
    monkeypatch.setattr(cli, "is_dev_install", lambda: True)
    monkeypatch.setattr(cli.subprocess, "run", MagicMock(return_value=MagicMock(returncode=7)))
    rc = cli.main(["--hardware", "cpu"])
    assert rc == 7
