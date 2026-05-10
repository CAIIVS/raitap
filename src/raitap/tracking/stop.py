"""``raitap tracking stop`` entry point.

Walks every loaded :class:`BaseTracker` subclass and invokes its
:meth:`~BaseTracker.stop_detached` hook. Each tracker owns its own shutdown
technique; this module only orchestrates the iteration and reports orphan
registry entries whose tracker class is no longer importable.
"""

from __future__ import annotations

from raitap import raitap_log

from .base_tracker import BaseTracker
from .process_registry import remaining_trackers


def run_stop_command() -> None:
    total_killed = 0
    total_skipped = 0

    try:
        for cls in BaseTracker.__subclasses__():
            killed, skipped = cls.stop_detached()
            total_killed += killed
            total_skipped += skipped
    except KeyboardInterrupt:
        raitap_log.warn(
            "Tracking stop interrupted by user after %d terminated, %d already dead",
            total_killed,
            total_skipped,
        )
        return

    orphans = sorted(remaining_trackers())
    if orphans:
        raitap_log.error(
            "Registry has entries for trackers not importable in this environment: %s. "
            "Install the matching extra or remove `~/.raitap/tracking_processes.json` "
            "to clear them.",
            ", ".join(orphans),
        )

    if total_killed == 0 and total_skipped == 0:
        raitap_log.info("No tracker background processes registered.")
        return
    raitap_log.info("Tracking stop: %d terminated, %d already dead", total_killed, total_skipped)


__all__ = ["run_stop_command"]
