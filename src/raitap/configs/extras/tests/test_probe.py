"""Tests for hardware probe — fully mocked, no real subprocess execution."""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import MagicMock

import pytest

from raitap.configs.extras import probe


@pytest.fixture(autouse=True)
def _reset_lru_cache() -> None:
    probe.detect_hardware.cache_clear()


def _stub_run(returncode: int = 0, stdout: str = "") -> MagicMock:
    return MagicMock(
        spec=subprocess.CompletedProcess, returncode=returncode, stdout=stdout, stderr=""
    )


def test_darwin_always_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "_platform", lambda: "darwin")
    assert probe.detect_hardware() == "cpu"


def test_cuda_when_nvidia_smi_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "_platform", lambda: "linux")
    monkeypatch.setattr(
        probe, "_which", lambda name: f"/usr/bin/{name}" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(probe, "_run", lambda *_a, **_k: _stub_run(returncode=0))
    assert probe.detect_hardware() == "cuda"


def test_nvidia_smi_present_but_failing_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "_platform", lambda: "linux")
    monkeypatch.setattr(
        probe, "_which", lambda name: f"/usr/bin/{name}" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(probe, "_run", lambda *_a, **_k: _stub_run(returncode=1))
    # No lspci configured; should fall back to cpu.
    assert probe.detect_hardware() == "cpu"


def test_intel_gpu_via_lspci_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "_platform", lambda: "linux")
    monkeypatch.setattr(
        probe, "_which", lambda name: f"/usr/bin/{name}" if name == "lspci" else None
    )

    def run(argv: list[str], **_k: Any) -> Any:
        if argv and argv[0].endswith("lspci"):
            return _stub_run(
                returncode=0, stdout="00:02.0 VGA compatible controller: Intel Corporation Arc A770"
            )
        return _stub_run(returncode=1)

    monkeypatch.setattr(probe, "_run", run)
    assert probe.detect_hardware() == "xpu"


def test_intel_gpu_via_powershell_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "_platform", lambda: "win32")
    monkeypatch.setattr(
        probe, "_which", lambda name: f"C:/{name}.exe" if name == "powershell" else None
    )

    def run(argv: list[str], **_k: Any) -> Any:
        if argv and argv[0].lower().endswith("powershell.exe"):
            return _stub_run(returncode=0, stdout="Name\n----\nIntel(R) Arc(TM) A770\n")
        return _stub_run(returncode=1)

    monkeypatch.setattr(probe, "_run", run)
    assert probe.detect_hardware() == "xpu"


def test_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "_platform", lambda: "linux")
    monkeypatch.setattr(probe, "_which", lambda _name: None)
    monkeypatch.setattr(probe, "_run", lambda *_a, **_k: _stub_run(returncode=1))
    assert probe.detect_hardware() == "cpu"


def test_subprocess_timeout_treated_as_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "_platform", lambda: "linux")
    monkeypatch.setattr(
        probe, "_which", lambda name: f"/usr/bin/{name}" if name == "nvidia-smi" else None
    )

    def raises(*_a: Any, **_k: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=3)

    monkeypatch.setattr(probe, "_run", raises)
    assert probe.detect_hardware() == "cpu"
