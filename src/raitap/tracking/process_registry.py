"""Cross-tracker storage of detached helper processes and watched ports.

Trackers that own long-lived helpers record them here so ``raitap tracking
stop`` can shut them down later. This module is the **data layer only** —
killing PIDs / resolving ports lives in :mod:`raitap.utils.process` and is
composed by each tracker's :meth:`BaseTracker.stop_detached` override.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from raitap import raitap_log
from raitap.utils.process import is_pid_alive

if TYPE_CHECKING:
    from collections.abc import Iterable

REGISTRY_PATH = Path.home() / ".raitap" / "tracking_processes.json"
STOP_COMMAND = "uv run raitap tracking stop"
_HINT_SHOWN = False


@dataclass(frozen=True)
class ProcessEntry:
    pid: int
    tracker: str
    label: str
    url: str | None
    started_at: float


@dataclass(frozen=True)
class WatchedPort:
    port: int
    host: str
    tracker: str
    label: str
    url: str | None
    started_at: float


def _load_state() -> tuple[list[ProcessEntry], list[WatchedPort]]:
    if not REGISTRY_PATH.exists():
        return [], []
    try:
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], []

    entries: list[ProcessEntry] = []
    for item in raw.get("entries", []):
        try:
            entries.append(
                ProcessEntry(
                    pid=int(item["pid"]),
                    tracker=str(item["tracker"]),
                    label=str(item["label"]),
                    url=item.get("url"),
                    started_at=float(item.get("started_at", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    ports: list[WatchedPort] = []
    for item in raw.get("watched_ports", []):
        try:
            ports.append(
                WatchedPort(
                    port=int(item["port"]),
                    host=str(item.get("host", "127.0.0.1")),
                    tracker=str(item["tracker"]),
                    label=str(item["label"]),
                    url=item.get("url"),
                    started_at=float(item.get("started_at", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    return entries, ports


def _save_state(entries: Iterable[ProcessEntry], ports: Iterable[WatchedPort]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": [asdict(e) for e in entries],
        "watched_ports": [asdict(p) for p in ports],
    }
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def register_background_process(
    pid: int,
    *,
    tracker: str,
    label: str,
    url: str | None = None,
    show_stop_hint: bool = True,
) -> None:
    """Record a tracker-spawned process and remind the user how to stop it."""
    global _HINT_SHOWN

    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return
    if pid_int <= 0:
        return
    pid = pid_int

    entries, ports = _load_state()
    entries = [e for e in entries if e.pid != pid and is_pid_alive(e.pid)]
    entries.append(
        ProcessEntry(
            pid=pid,
            tracker=tracker,
            label=label,
            url=url,
            started_at=time.time(),
        )
    )
    _save_state(entries, ports)

    if show_stop_hint and not _HINT_SHOWN:
        raitap_log.info(
            "%s is detached and survives this run. Stop it with: `%s`",
            label,
            STOP_COMMAND,
        )
        _HINT_SHOWN = True


def watch_port(
    port: int,
    *,
    host: str = "127.0.0.1",
    tracker: str,
    label: str,
    url: str | None = None,
) -> None:
    """Register a localhost port a tracker reuses but did not spawn."""
    entries, ports = _load_state()
    ports = [p for p in ports if not (p.port == port and p.host == host)]
    ports.append(
        WatchedPort(
            port=port,
            host=host,
            tracker=tracker,
            label=label,
            url=url,
            started_at=time.time(),
        )
    )
    _save_state(entries, ports)


def announce_stop_hint() -> None:
    """Print the stop hint once per process. Idempotent across multiple calls."""
    global _HINT_SHOWN
    if _HINT_SHOWN:
        return
    raitap_log.info("Stop detached tracker processes any time with: `%s`", STOP_COMMAND)
    _HINT_SHOWN = True


def list_active() -> list[ProcessEntry]:
    entries, ports = _load_state()
    alive = [e for e in entries if is_pid_alive(e.pid)]
    if len(alive) != len(entries):
        _save_state(alive, ports)
    return alive


def pop_entries_for_tracker(
    tracker: str,
) -> tuple[list[ProcessEntry], list[WatchedPort]]:
    """Remove and return all registry entries belonging to ``tracker``."""
    entries, ports = _load_state()
    mine_e = [e for e in entries if e.tracker == tracker]
    mine_p = [p for p in ports if p.tracker == tracker]
    rest_e = [e for e in entries if e.tracker != tracker]
    rest_p = [p for p in ports if p.tracker != tracker]
    _save_state(rest_e, rest_p)
    return mine_e, mine_p


def reinsert_entries(entries: Iterable[ProcessEntry], ports: Iterable[WatchedPort]) -> None:
    """Put failed entries back into the registry so a later stop can retry."""
    existing_e, existing_p = _load_state()
    _save_state([*existing_e, *entries], [*existing_p, *ports])


def remaining_trackers() -> set[str]:
    """Distinct tracker names that still have entries."""
    entries, ports = _load_state()
    return {e.tracker for e in entries} | {p.tracker for p in ports}


__all__ = [
    "REGISTRY_PATH",
    "STOP_COMMAND",
    "ProcessEntry",
    "WatchedPort",
    "announce_stop_hint",
    "list_active",
    "pop_entries_for_tracker",
    "register_background_process",
    "reinsert_entries",
    "remaining_trackers",
    "watch_port",
]
