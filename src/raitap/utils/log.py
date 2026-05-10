"""Unified raitap logger: ``raitap_log.warn`` / ``.info`` / ``.debug`` / ``.error`` / ``.suppress``.

A single import surface so call sites never have to choose between
:func:`warnings.warn`, :func:`logging.info`, ``logging.getLogger(__name__)``, and
``raise``. The methods underneath dispatch to the right primitive:

- :meth:`_RaitapLog.warn` → :func:`warnings.warn` (preserves ``filterwarnings``,
  ``catch_warnings``, ``pytest.warns``).
- :meth:`_RaitapLog.info` / :meth:`.debug` / :meth:`.error` / :meth:`.critical`
  → :class:`logging.Logger` of the **caller's module** (resolved via
  ``sys._getframe``), so per-module log levels configured through
  :func:`logging.dictConfig` keep working — call sites never have to
  ``logger = logging.getLogger(__name__)`` themselves.
- :meth:`_RaitapLog.suppress` → :func:`warnings.filterwarnings` shortcut for
  silencing known-noise library warnings at adapter import time.

Raising errors is **not** part of this facade. Use ``raise SomeException(...)``
directly. Issue #22 introduces raitap exception classes that carry a
:class:`~raitap.utils.diagnostics.Diagnostic` payload; until then, plain
``raise`` is the right tool.

The thread-local diagnostic queue is the bridge from
:func:`warnings.formatwarning` (frames at warn time) to the rich log handler
(panels at emit time, after frames are unwound). It lives here because it is
warning-specific glue.
"""

from __future__ import annotations

import logging
import sys
import threading
import warnings
from collections import deque
from typing import Any

from raitap.utils.diagnostics import Diagnostic, Subsystem

_FALLBACK_LOGGER_NAME = "raitap"

# Per-thread FIFO of diagnostics resolved at ``warnings.formatwarning`` time,
# drained by the rich handler when it parses a warning panel. Thread-local so
# concurrent ``warnings.warn`` calls (e.g. dataloader workers) can't shuffle
# diagnostics between unrelated panels.
_diagnostic_state = threading.local()
_diagnostic_override = threading.local()
_diagnostic_override.value = None


def _diagnostic_queue() -> deque[Diagnostic]:
    queue = getattr(_diagnostic_state, "queue", None)
    if queue is None:
        queue = deque()
        _diagnostic_state.queue = queue
    return queue


def _caller_logger(stacklevel: int) -> logging.Logger:
    """Return the :class:`logging.Logger` for the caller's module.

    ``stacklevel`` follows :mod:`logging` conventions: ``2`` blames the
    immediate caller of the public ``raitap_log`` method that invoked us.
    """
    frame = sys._getframe(max(stacklevel, 0))
    module_name = frame.f_globals.get("__name__") or _FALLBACK_LOGGER_NAME
    return logging.getLogger(module_name)


class _RaitapLog:
    """Singleton facade. Import the module-level :data:`raitap_log` instance."""

    def warn(
        self,
        message: str,
        *args: object,
        subsystem: Subsystem | None = None,
        third_party_lib: str | None = None,
        category: type[Warning] = UserWarning,
        stacklevel: int = 2,
    ) -> None:
        """Emit a warning, optionally tagged with a logical raitap subsystem.

        Single warning verb in the facade — operational and user-facing
        warnings both go through here. Routed via :func:`warnings.warn` so the
        rich handler frames it as a panel with subsystem chip + docs link, and
        :func:`logging.captureWarnings` (installed by ``setup_logging``)
        forwards it to the logging system so MLflow / Airflow / any handler
        attached to the root logger picks it up too.

        ``*args`` enables printf-style formatting matching :class:`logging.Logger`:
        ``raitap_log.warn("loaded %d samples from %s", n, path)``.

        When ``subsystem`` is omitted the rich handler walks frames to classify
        the origin — usually correct when the call site lives inside the right
        subsystem directory. Pass ``subsystem`` explicitly only when the
        *logical* subsystem differs from the file path (e.g. ``run/pipeline.py``
        emitting a robustness warning).

        ``stacklevel`` follows :func:`warnings.warn` semantics: ``2`` (default)
        blames the immediate caller of this method.
        """
        if subsystem is not None:
            frame = sys._getframe(max(stacklevel - 1, 0))
            _diagnostic_override.value = Diagnostic(
                subsystem=subsystem,
                file=frame.f_code.co_filename,
                line=frame.f_lineno,
                third_party_lib=third_party_lib,
            )
        formatted = message % args if args else message
        try:
            warnings.warn(formatted, category, stacklevel=stacklevel)
        finally:
            _diagnostic_override.value = None

    def info(self, message: object, *args: object, stacklevel: int = 2, **kwargs: Any) -> None:
        """Log at INFO level on the caller's module logger."""
        _caller_logger(stacklevel).info(message, *args, stacklevel=stacklevel, **kwargs)

    def debug(self, message: object, *args: object, stacklevel: int = 2, **kwargs: Any) -> None:
        """Log at DEBUG level on the caller's module logger."""
        _caller_logger(stacklevel).debug(message, *args, stacklevel=stacklevel, **kwargs)

    def error(self, message: object, *args: object, stacklevel: int = 2, **kwargs: Any) -> None:
        """Log at ERROR level on the caller's module logger.

        Does **not** raise — for raising, use ``raise SomeException(...)``.
        Future raitap exceptions (issue #22) will carry a diagnostic payload.
        """
        _caller_logger(stacklevel).error(message, *args, stacklevel=stacklevel, **kwargs)

    def critical(self, message: object, *args: object, stacklevel: int = 2, **kwargs: Any) -> None:
        """Log at CRITICAL level on the caller's module logger."""
        _caller_logger(stacklevel).critical(message, *args, stacklevel=stacklevel, **kwargs)

    def exception(self, message: object, *args: object, stacklevel: int = 2, **kwargs: Any) -> None:
        """Log at ERROR level with traceback. Use inside an ``except`` block."""
        _caller_logger(stacklevel).exception(message, *args, stacklevel=stacklevel, **kwargs)

    def suppress(
        self,
        *,
        message: str,
        category: type[Warning] = UserWarning,
        module: str = "",
    ) -> None:
        """Register a runtime ``warnings.filterwarnings("ignore", ...)`` filter.

        Adapters call this at import time to silence known-noise warnings from
        wrapped libraries. Match arguments mirror :func:`warnings.filterwarnings`.
        """
        warnings.filterwarnings("ignore", message=message, category=category, module=module)


raitap_log = _RaitapLog()


def _take_diagnostic_override() -> Diagnostic | None:
    """Pop the current thread's diagnostic override, if any."""
    override = getattr(_diagnostic_override, "value", None)
    _diagnostic_override.value = None
    return override


def _push_diagnostic(diagnostic: Diagnostic) -> None:
    """Stash a diagnostic for the rich handler to drain."""
    _diagnostic_queue().append(diagnostic)


def _pop_diagnostic() -> Diagnostic | None:
    """Pop the next stashed diagnostic, or ``None`` if the queue is empty."""
    queue = _diagnostic_queue()
    if not queue:
        return None
    return queue.popleft()


def _clear_diagnostics() -> None:
    """Test helper: reset the pending-diagnostic queue for the current thread."""
    _diagnostic_queue().clear()


__all__ = ["raitap_log"]
