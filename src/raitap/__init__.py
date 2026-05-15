"""Top-level public surface for raitap.

Re-exports the unified :data:`raitap_log` singleton so call sites can use one
import for warnings, info logs, and (future) errors:

    from raitap import raitap_log

    raitap_log.warn("…")
    raitap_log.info("…")

Also re-exports the programmatic entry point :func:`run` and the :class:`AppConfig`
dataclass so library users can drive the pipeline without touching Hydra:

    from raitap import run, AppConfig
"""

from __future__ import annotations

# Bind ``raitap_log`` BEFORE importing ``raitap.api``. Downstream modules
# (e.g. ``raitap.pipeline.orchestrator``) do ``from raitap import raitap_log``
# at module-import time, so the attribute must exist on the package namespace
# before any of that chain is touched. The ordering is load-bearing — do
# not let isort/ruff reorder it.
from raitap.utils.log import raitap_log  # isort: skip
from raitap.api import AppConfig, run

__all__ = ["AppConfig", "raitap_log", "run"]
