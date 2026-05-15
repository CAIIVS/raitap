"""Top-level public surface for raitap.

Re-exports the unified :data:`raitap_log` singleton so call sites can use one
import for warnings, info logs, and (future) errors::

    from raitap import raitap_log

    raitap_log.warn("…")
    raitap_log.info("…")

Also re-exports the programmatic entry point :func:`run` and the
:class:`AppConfig` dataclass so library users can drive the pipeline without
touching Hydra::

    from raitap import run, AppConfig

``run`` and ``AppConfig`` are loaded lazily via :pep:`562` ``__getattr__`` to
keep ``import raitap`` lightweight: the underlying :mod:`raitap.api` module
imports torchmetrics / Captum / torchattacks / etc., which a bare-bones
``pip install raitap`` does not pull in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.utils.log import raitap_log

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.pipeline.outputs import RunOutputs

    def run(config: AppConfig, *, verbose: bool = True) -> RunOutputs: ...


__all__ = ["AppConfig", "raitap_log", "run"]


def __getattr__(name: str) -> Any:
    if name in {"run", "AppConfig"}:
        from raitap.api import AppConfig as _AppConfig
        from raitap.api import run as _run

        return {"run": _run, "AppConfig": _AppConfig}[name]
    raise AttributeError(f"module 'raitap' has no attribute {name!r}")
