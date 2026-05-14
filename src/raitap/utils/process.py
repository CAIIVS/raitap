"""Generic, cross-platform process / port primitives.

Tracker shutdown code composes these helpers. The module is deliberately
free of tracker-specific knowledge so any module that needs to terminate
local helpers can reuse it.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def is_port_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def pids_listening_on_port(port: int) -> list[int]:
    """PIDs of processes listening on ``port``. Empty if none or tools missing."""
    if sys.platform == "win32":
        return _pids_listening_windows(port)
    return _pids_listening_unix(port)


def child_pids(pid: int) -> list[int]:
    """Direct children of ``pid`` via OS-native tooling. Best-effort."""
    if sys.platform == "win32":
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if pwsh is None:
            return []
        cmd = [
            pwsh,
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process "
                f"-Filter 'ParentProcessId={pid}' | "
                "Select-Object -ExpandProperty ProcessId"
            ),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        return _parse_pid_lines(result.stdout)

    ps = shutil.which("ps")
    if ps is None:
        return []
    result = subprocess.run(
        [ps, "-o", "pid=", "--ppid", str(pid)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return _parse_pid_lines(result.stdout)


def descendant_pids(pid: int) -> list[int]:
    """Breadth-first walk of the process tree rooted at ``pid``."""
    descendants: list[int] = []
    seen: set[int] = {pid}
    queue = [pid]
    while queue:
        current = queue.pop(0)
        for child in child_pids(current):
            if child in seen:
                continue
            seen.add(child)
            descendants.append(child)
            queue.append(child)
    return descendants


def terminate_pid_tree(pid: int, *, timeout: float = 5.0) -> bool:
    """SIGTERM the process tree rooted at ``pid``, then SIGKILL stragglers.

    Returns ``True`` if the root PID is no longer alive when this call returns.
    Callers decide how to surface partial failures — this helper stays quiet so
    side-effect checks (e.g. port no longer listening) can rescue stale liveness
    reads on Windows.
    """
    if sys.platform == "win32":
        # ``taskkill /F /T`` walks the live process tree at kernel level, which
        # catches multiprocessing workers that re-parent after the root dies
        # (a Python ``os.kill`` loop misses those because the snapshot is
        # stale by the time SIGTERM lands).
        taskkill = shutil.which("taskkill")
        if taskkill is not None:
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                subprocess.run(
                    [taskkill, "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
            return not is_pid_alive(pid)

    descendants = descendant_pids(pid)
    targets = [pid, *descendants]

    for target in targets:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(target, signal.SIGTERM)

    deadline = time.time() + timeout
    try:
        while time.time() < deadline and any(is_pid_alive(t) for t in targets):
            time.sleep(0.1)
    except KeyboardInterrupt:
        # User aborted mid-wait. Skip the grace period and go straight to
        # SIGKILL so we don't leave processes hanging.
        pass

    for target in targets:
        if is_pid_alive(target):
            with contextlib.suppress(OSError):
                os.kill(target, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)

    return not any(is_pid_alive(t) for t in targets)


def _pids_listening_windows(port: int) -> list[int]:
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is not None:
        cmd = [
            pwsh,
            "-NoProfile",
            "-Command",
            (
                f"Get-NetTCPConnection -LocalPort {port} -State Listen "
                "-ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"
            ),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        pids = _parse_pid_lines(result.stdout)
        if pids:
            return pids

    netstat = shutil.which("netstat")
    if netstat is not None:
        result = subprocess.run(
            [netstat, "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        pids: list[int] = []
        pattern = re.compile(rf":\s*{port}\s+\S+\s+LISTENING\s+(\d+)")
        for line in result.stdout.splitlines():
            match = pattern.search(line)
            if match:
                pids.append(int(match.group(1)))
        return pids
    return []


def _pids_listening_unix(port: int) -> list[int]:
    lsof = shutil.which("lsof")
    if lsof is not None:
        result = subprocess.run(
            [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        pids = _parse_pid_lines(result.stdout)
        if pids:
            return pids

    ss = shutil.which("ss")
    if ss is not None:
        result = subprocess.run(
            [ss, "-ltnpH", f"sport = :{port}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return [int(m) for m in re.findall(r"pid=(\d+)", result.stdout)]
    return []


def _parse_pid_lines(text: str) -> list[int]:
    pids: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


__all__ = [
    "child_pids",
    "descendant_pids",
    "is_pid_alive",
    "is_port_listening",
    "pids_listening_on_port",
    "terminate_pid_tree",
]
