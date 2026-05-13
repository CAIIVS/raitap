"""Host hardware detection for raitap-deps.

Returns one of ``cpu``, ``cuda``, ``xpu``. ``cuda`` requires a working
``nvidia-smi``; ``xpu`` requires an Intel GPU visible to the platform-specific
enumeration command. All subprocess calls are wrapped in thin helpers so the
test suite can monkey-patch them without touching the real ``subprocess``
module.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Literal

Hardware = Literal["cpu", "cuda", "xpu"]


def _platform() -> str:
    return sys.platform


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(argv: list[str], *, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def _cuda_available() -> bool:
    if _which("nvidia-smi") is None:
        return False
    try:
        result = _run(["nvidia-smi"], timeout=3.0)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _intel_gpu_linux() -> bool:
    if _which("lspci") is None:
        return False
    try:
        result = _run(["lspci"], timeout=3.0)
    except (subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode != 0:
        return False
    return any(
        ("Intel" in line) and any(tag in line for tag in ("VGA", "3D", "Display"))
        for line in result.stdout.splitlines()
    )


def _intel_gpu_windows() -> bool:
    powershell = _which("powershell")
    if powershell is None:
        return False
    try:
        result = _run(
            [
                powershell,
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            timeout=5.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode != 0:
        return False
    return any("Intel" in line for line in result.stdout.splitlines())


@lru_cache(maxsize=1)
def detect_hardware() -> Hardware:
    """Return the best hardware extra suffix for the current host."""
    platform = _platform()
    if platform == "darwin":
        return "cpu"
    if _cuda_available():
        return "cuda"
    if platform.startswith("linux") and _intel_gpu_linux():
        return "xpu"
    if platform.startswith("win") and _intel_gpu_windows():
        return "xpu"
    return "cpu"
