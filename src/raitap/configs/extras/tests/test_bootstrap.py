"""Tests for the raitap deps-bootstrap entry point."""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import MagicMock

import pytest

from raitap.configs.extras import bootstrap


@pytest.fixture(autouse=True)
def _isolate_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(bootstrap._SENTINEL, raising=False)


def _fake_compose(monkeypatch: pytest.MonkeyPatch, cfg: dict[str, Any]) -> None:
    monkeypatch.setattr(bootstrap, "_compose", lambda *_a, **_k: cfg)


def _baseline_cfg() -> dict[str, Any]:
    return {
        "model": {"source": "x.pt"},
        "metrics": {"_target_": "ClassificationMetrics", "task": "multiclass"},
        "reporting": {"_target_": "HTMLReporter", "filename": "r"},
    }


def test_sentinel_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(bootstrap._SENTINEL, "1")
    run_mock = MagicMock()
    monkeypatch.setattr(bootstrap.subprocess, "run", run_mock)
    cleaned = bootstrap.maybe_bootstrap(["raitap", "data=mnist_samples"])
    assert cleaned == ["raitap", "data=mnist_samples"]
    run_mock.assert_not_called()


def test_custom_deps_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    run_mock = MagicMock()
    monkeypatch.setattr(bootstrap.subprocess, "run", run_mock)
    cleaned = bootstrap.maybe_bootstrap(["raitap", "--custom-deps", "data=mnist_samples"])
    assert cleaned == ["raitap", "data=mnist_samples"]
    run_mock.assert_not_called()


def test_dry_run_prints_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_compose(monkeypatch, _baseline_cfg())
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: "cpu")
    monkeypatch.setattr(bootstrap, "pick_python_version", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "check_platform_availability", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "validate_conflicts", lambda *_a, **_k: None)
    run_mock = MagicMock()
    monkeypatch.setattr(bootstrap.subprocess, "run", run_mock)
    with pytest.raises(SystemExit) as excinfo:
        bootstrap.maybe_bootstrap(["raitap", "--dry-run"])
    assert excinfo.value.code == 0
    run_mock.assert_not_called()
    captured = capsys.readouterr().out
    assert "Deps" in captured
    assert "Dry-run preview" in captured
    assert "torch-cpu" in captured


def test_default_relaunches_via_uv_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_compose(monkeypatch, _baseline_cfg())
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: "cpu")
    monkeypatch.setattr(bootstrap, "pick_python_version", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "check_platform_availability", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "validate_conflicts", lambda *_a, **_k: None)
    run_mock = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(bootstrap.subprocess, "run", run_mock)
    with pytest.raises(SystemExit) as excinfo:
        bootstrap.maybe_bootstrap(["raitap", "data=mnist_samples"])
    assert excinfo.value.code == 0
    argv = run_mock.call_args.args[0]
    assert argv[:2] == ["uv", "run"]
    assert "--extra" in argv
    assert "torch-cpu" in argv
    assert argv[-2:] == ["raitap", "data=mnist_samples"]
    env = run_mock.call_args.kwargs["env"]
    assert env[bootstrap._SENTINEL] == "1"


def test_python_pin_inserted(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_compose(monkeypatch, _baseline_cfg())
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: "cpu")
    monkeypatch.setattr(bootstrap, "pick_python_version", lambda *_a, **_k: "3.11")
    monkeypatch.setattr(bootstrap, "check_platform_availability", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "validate_conflicts", lambda *_a, **_k: None)
    run_mock = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(bootstrap.subprocess, "run", run_mock)
    with pytest.raises(SystemExit):
        bootstrap.maybe_bootstrap(["raitap"])
    argv = run_mock.call_args.args[0]
    assert argv[:4] == ["uv", "run", "-p", "3.11"]


def test_unavailable_extra_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_compose(monkeypatch, _baseline_cfg())
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: "cpu")

    def boom(*_a: Any, **_k: Any) -> None:
        raise bootstrap.ExtraUnavailableError("nope")

    monkeypatch.setattr(bootstrap, "check_platform_availability", boom)
    monkeypatch.setattr(bootstrap, "validate_conflicts", lambda *_a, **_k: None)
    with pytest.raises(SystemExit) as excinfo:
        bootstrap.maybe_bootstrap(["raitap"])
    assert excinfo.value.code == 2


def test_relaunch_propagates_child_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_compose(monkeypatch, _baseline_cfg())
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: "cpu")
    monkeypatch.setattr(bootstrap, "pick_python_version", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "check_platform_availability", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "validate_conflicts", lambda *_a, **_k: None)
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=7)),
    )
    with pytest.raises(SystemExit) as excinfo:
        bootstrap.maybe_bootstrap(["raitap"])
    assert excinfo.value.code == 7


def test_strip_deps_flags() -> None:
    cleaned, dry, sync_only, custom = bootstrap._strip_deps_flags(
        ["raitap", "--dry-run", "--sync-only", "--custom-deps", "data=x"]
    )
    assert cleaned == ["raitap", "data=x"]
    assert dry is True
    assert sync_only is True
    assert custom is True


def test_sync_only_runs_sync_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_compose(monkeypatch, _baseline_cfg())
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: "cpu")
    monkeypatch.setattr(bootstrap, "pick_python_version", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "check_platform_availability", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "validate_conflicts", lambda *_a, **_k: None)
    run_mock = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(bootstrap.subprocess, "run", run_mock)
    with pytest.raises(SystemExit) as excinfo:
        bootstrap.maybe_bootstrap(["raitap", "--sync-only"])
    assert excinfo.value.code == 0
    argv = run_mock.call_args.args[0]
    assert argv[:2] == ["uv", "sync"]
    # No re-launch as `raitap`; no sentinel env.
    assert "raitap" not in argv


def test_hydra_overrides_skips_config_flags() -> None:
    argv = ["--config-dir", "/tmp/x", "--config-name", "config", "data=x", "model=y"]
    assert bootstrap._hydra_overrides(argv) == ["data=x", "model=y"]


def test_hydra_overrides_skips_eq_form() -> None:
    argv = ["--config-dir=/tmp/x", "--config-name=config", "data=x"]
    assert bootstrap._hydra_overrides(argv) == ["data=x"]
