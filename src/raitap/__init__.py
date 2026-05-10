"""Top-level public surface for raitap.

Re-exports the unified :data:`raitap_log` singleton so call sites can use one
import for warnings, info logs, and (future) errors:

    from raitap import raitap_log

    raitap_log.warn("…")
    raitap_log.info("…")
"""

from __future__ import annotations

from raitap.utils.log import raitap_log

__all__ = ["raitap_log"]
