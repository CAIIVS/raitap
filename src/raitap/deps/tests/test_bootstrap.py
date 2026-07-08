"""Tests for the raitap deps-bootstrap entry point."""

from __future__ import annotations

import os
import subprocess
from typing import Any
from unittest.mock import MagicMock

import pytest

from raitap.deps import bootstrap
from raitap.types import ResolvedHardware


@pytest.fixture(autouse=True)
def _isolate_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(bootstrap._SENTINEL, raising=False)


def _fake_compose(monkeypatch: pytest.MonkeyPatch, cfg: dict[str, Any]) -> None:
    monkeypatch.setattr(bootstrap, "_compose", lambda *_a, **_k: cfg)


def _baseline_cfg() -> dict[str, Any]:
    return {
        "model": {"source": "x.pt"},
        "metrics": {"use": "multiclass_classification", "num_classes": 3},
        "reporting": {"use": "html", "filename": "r"},
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


def _stub_common(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_compose(monkeypatch, _baseline_cfg())
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: ResolvedHardware.cpu)
    monkeypatch.setattr(bootstrap, "pick_python_version", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "check_platform_availability", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "validate_conflicts", lambda *_a, **_k: None)


def test_case_b_no_uv_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_common(monkeypatch)
    monkeypatch.setattr(bootstrap, "_is_dev_install", lambda: True)
    monkeypatch.setattr(bootstrap, "_uv_available", lambda: False)
    with pytest.raises(SystemExit) as excinfo:
        bootstrap.maybe_bootstrap(["raitap"])
    assert excinfo.value.code == 2


def test_case_c_without_consent_shows_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_common(monkeypatch)
    monkeypatch.setattr(bootstrap, "_is_dev_install", lambda: False)
    monkeypatch.setattr(bootstrap, "_uv_available", lambda: True)
    run_mock = MagicMock()
    monkeypatch.setattr(bootstrap.subprocess, "run", run_mock)
    with pytest.raises(SystemExit) as excinfo:
        bootstrap.maybe_bootstrap(["raitap"])
    assert excinfo.value.code == 1
    run_mock.assert_not_called()


def test_case_c_with_consent_runs_uv_add(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_common(monkeypatch)
    monkeypatch.setattr(bootstrap, "_is_dev_install", lambda: False)
    monkeypatch.setattr(bootstrap, "_uv_available", lambda: True)
    run_mock = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(bootstrap.subprocess, "run", run_mock)
    with pytest.raises(SystemExit):
        bootstrap.maybe_bootstrap(["raitap", "--allow-project-edit"])
    first_call = run_mock.call_args_list[0].args[0]
    assert first_call[:2] == ["uv", "add"]
    assert any("raitap[" in a for a in first_call)


def test_case_d_in_venv_runs_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_common(monkeypatch)
    monkeypatch.setattr(bootstrap, "_is_dev_install", lambda: False)
    monkeypatch.setattr(bootstrap, "_uv_available", lambda: False)
    monkeypatch.setattr(bootstrap, "_in_venv", lambda: True)
    run_mock = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(bootstrap.subprocess, "run", run_mock)
    with pytest.raises(SystemExit):
        bootstrap.maybe_bootstrap(["raitap"])
    first_call = run_mock.call_args_list[0].args[0]
    assert first_call[1:4] == ["-m", "pip", "install"]


def test_case_d_global_without_consent_shows_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_common(monkeypatch)
    monkeypatch.setattr(bootstrap, "_is_dev_install", lambda: False)
    monkeypatch.setattr(bootstrap, "_uv_available", lambda: False)
    monkeypatch.setattr(bootstrap, "_in_venv", lambda: False)
    run_mock = MagicMock()
    monkeypatch.setattr(bootstrap.subprocess, "run", run_mock)
    with pytest.raises(SystemExit) as excinfo:
        bootstrap.maybe_bootstrap(["raitap"])
    assert excinfo.value.code == 1
    run_mock.assert_not_called()


def test_case_d_global_with_exec_flag_runs_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_common(monkeypatch)
    monkeypatch.setattr(bootstrap, "_is_dev_install", lambda: False)
    monkeypatch.setattr(bootstrap, "_uv_available", lambda: False)
    monkeypatch.setattr(bootstrap, "_in_venv", lambda: False)
    run_mock = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(bootstrap.subprocess, "run", run_mock)
    with pytest.raises(SystemExit):
        bootstrap.maybe_bootstrap(["raitap", "--exec-global"])
    first_call = run_mock.call_args_list[0].args[0]
    assert first_call[1:4] == ["-m", "pip", "install"]


def test_python_pin_mismatch_aborts_non_uv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_common(monkeypatch)
    monkeypatch.setattr(bootstrap, "_is_dev_install", lambda: False)
    monkeypatch.setattr(bootstrap, "_uv_available", lambda: True)
    monkeypatch.setattr(bootstrap, "pick_python_version", lambda *_a, **_k: "3.99")
    with pytest.raises(SystemExit) as excinfo:
        bootstrap.maybe_bootstrap(["raitap", "--allow-project-edit"])
    assert excinfo.value.code == 2


def test_dry_run_prints_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_compose(monkeypatch, _baseline_cfg())
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: ResolvedHardware.cpu)
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
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: ResolvedHardware.cpu)
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
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: ResolvedHardware.cpu)
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
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: ResolvedHardware.cpu)

    def boom(*_a: Any, **_k: Any) -> None:
        raise bootstrap.ExtraUnavailableError("nope")

    monkeypatch.setattr(bootstrap, "check_platform_availability", boom)
    monkeypatch.setattr(bootstrap, "validate_conflicts", lambda *_a, **_k: None)
    with pytest.raises(SystemExit) as excinfo:
        bootstrap.maybe_bootstrap(["raitap"])
    assert excinfo.value.code == 2


def test_relaunch_propagates_child_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_compose(monkeypatch, _baseline_cfg())
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: ResolvedHardware.cpu)
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
    cleaned, flags = bootstrap._strip_deps_flags(
        [
            "raitap",
            "--dry-run",
            "--sync-only",
            "--custom-deps",
            "--allow-project-edit",
            "--exec-global",
            "data=x",
        ]
    )
    assert cleaned == ["raitap", "data=x"]
    assert flags.dry_run is True
    assert flags.sync_only is True
    assert flags.custom is True
    assert flags.allow_project_edit is True
    assert flags.exec_global is True


def test_strip_deps_flags_accepts_y_alias() -> None:
    cleaned, flags = bootstrap._strip_deps_flags(["raitap", "-y", "data=x"])
    assert cleaned == ["raitap", "data=x"]
    assert flags.allow_project_edit is True


def test_sync_only_runs_sync_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_compose(monkeypatch, _baseline_cfg())
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: ResolvedHardware.cpu)
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


def test_strip_deps_flags_strips_allow_preprocessing_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAITAP_ALLOW_PREPROCESSING_EXEC", raising=False)
    cleaned, flags = bootstrap._strip_deps_flags(["raitap", "--demo", "--allow-preprocessing-exec"])
    assert cleaned == ["raitap", "--demo"]
    assert flags.allow_preprocessing_exec is True


def test_strip_deps_flags_strips_yp_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAITAP_ALLOW_PREPROCESSING_EXEC", raising=False)
    cleaned, flags = bootstrap._strip_deps_flags(["raitap", "--demo", "-yp"])
    assert cleaned == ["raitap", "--demo"]
    assert flags.allow_preprocessing_exec is True


def test_strip_deps_flags_exports_preprocessing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAITAP_ALLOW_PREPROCESSING_EXEC", raising=False)
    bootstrap._strip_deps_flags(["raitap", "-yp"])
    assert os.environ["RAITAP_ALLOW_PREPROCESSING_EXEC"] == "1"


def test_strip_deps_flags_does_not_export_when_flag_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAITAP_ALLOW_PREPROCESSING_EXEC", raising=False)
    bootstrap._strip_deps_flags(["raitap", "--demo"])
    assert "RAITAP_ALLOW_PREPROCESSING_EXEC" not in os.environ


def test_strip_deps_flags_combines_y_and_yp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAITAP_ALLOW_PREPROCESSING_EXEC", raising=False)
    cleaned, flags = bootstrap._strip_deps_flags(["raitap", "-y", "-yp"])
    assert cleaned == ["raitap"]
    assert flags.allow_project_edit is True
    assert flags.allow_preprocessing_exec is True


def test_strip_deps_flags_strips_acknowledge_preprocessing_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAITAP_ACKNOWLEDGE_PREPROCESSING_OFF", raising=False)
    cleaned, flags = bootstrap._strip_deps_flags(
        ["raitap", "--demo", "--acknowledge-preprocessing-off"]
    )
    assert cleaned == ["raitap", "--demo"]
    assert flags.acknowledge_preprocessing_off is True


def test_strip_deps_flags_exports_acknowledge_off_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAITAP_ACKNOWLEDGE_PREPROCESSING_OFF", raising=False)
    bootstrap._strip_deps_flags(["raitap", "--acknowledge-preprocessing-off"])
    assert os.environ["RAITAP_ACKNOWLEDGE_PREPROCESSING_OFF"] == "1"


def test_strip_deps_flags_does_not_export_acknowledge_off_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAITAP_ACKNOWLEDGE_PREPROCESSING_OFF", raising=False)
    bootstrap._strip_deps_flags(["raitap", "--demo"])
    assert "RAITAP_ACKNOWLEDGE_PREPROCESSING_OFF" not in os.environ


def test_strip_deps_flags_strips_allow_unsafe_pickle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAITAP_ALLOW_UNSAFE_PICKLE", raising=False)
    cleaned, flags = bootstrap._strip_deps_flags(["raitap", "--demo", "--allow-unsafe-pickle"])
    assert cleaned == ["raitap", "--demo"]
    assert flags.allow_unsafe_pickle is True


def test_strip_deps_flags_exports_allow_unsafe_pickle_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAITAP_ALLOW_UNSAFE_PICKLE", raising=False)
    bootstrap._strip_deps_flags(["raitap", "--allow-unsafe-pickle"])
    try:
        assert os.environ["RAITAP_ALLOW_UNSAFE_PICKLE"] == "1"
    finally:
        # ``_strip_deps_flags`` mutates ``os.environ`` directly; monkeypatch
        # only tracks values it set, so clean up here to avoid leaking the
        # consent into pickle-refusal tests elsewhere in the suite.
        os.environ.pop("RAITAP_ALLOW_UNSAFE_PICKLE", None)


def test_strip_deps_flags_does_not_export_allow_unsafe_pickle_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAITAP_ALLOW_UNSAFE_PICKLE", raising=False)
    bootstrap._strip_deps_flags(["raitap", "--demo"])
    assert "RAITAP_ALLOW_UNSAFE_PICKLE" not in os.environ
