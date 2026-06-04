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
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from raitap.utils.diagnostics import Diagnostic, Module

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

    def __init__(self) -> None:
        # When inside a ``deferred()`` block this holds buffered INFO/DEBUG
        # records ``(logger, level, message, args, kwargs)`` to replay on exit;
        # ``None`` means "emit immediately". Errors/criticals are never deferred.
        self._deferred_records: list[tuple[logging.Logger, int, object, tuple, dict]] | None = None

    def warn(
        self,
        message: str,
        *args: object,
        module: Module | None = None,
        third_party_lib: str | None = None,
        category: type[Warning] = UserWarning,
        stacklevel: int = 2,
    ) -> None:
        """Emit a warning, optionally tagged with a logical raitap module.

        Single warning verb in the facade — operational and user-facing
        warnings both go through here. Routed via :func:`warnings.warn` so the
        rich handler frames it as a panel with module chip + docs link, and
        :func:`logging.captureWarnings` (installed by ``setup_logging``)
        forwards it to the logging system so MLflow / Airflow / any handler
        attached to the root logger picks it up too.

        ``*args`` enables printf-style formatting matching :class:`logging.Logger`:
        ``raitap_log.warn("loaded %d samples from %s", n, path)``.

        When ``module`` is omitted the rich handler walks frames to classify
        the origin — usually correct when the call site lives inside the right
        module directory. Pass ``module`` explicitly only when the
        *logical* module differs from the file path (e.g. ``run/pipeline.py``
        emitting a robustness warning).

        ``stacklevel`` follows :func:`warnings.warn` semantics: ``2`` (default)
        blames the immediate caller of this method.
        """
        if module is not None:
            frame = sys._getframe(max(stacklevel - 1, 0))
            _diagnostic_override.value = Diagnostic(
                module=module,
                file=frame.f_code.co_filename,
                line=frame.f_lineno,
                third_party_lib=third_party_lib,
            )
        formatted = message % args if args else message
        try:
            warnings.warn(formatted, category, stacklevel=stacklevel)
        finally:
            _diagnostic_override.value = None

    @staticmethod
    def _with_module(kwargs: dict[str, Any], module: Module | str | None) -> dict[str, Any]:
        """Stash an explicit module on the record (``extra``) so the rich handler
        renders its chip even when the *emitting* file differs from the logical
        module (e.g. the shared ``run_adapters`` loop logging for robustness)."""
        if module is None:
            return kwargs
        extra = {**(kwargs.get("extra") or {}), "_raitap_module": str(module)}
        return {**kwargs, "extra": extra}

    def info(
        self,
        message: object,
        *args: object,
        stacklevel: int = 2,
        module: Module | None = None,
        **kwargs: Any,
    ) -> None:
        """Log at INFO level on the caller's module logger.

        ``module`` overrides the rich handler's module-chip classification (which
        otherwise infers from the caller's logger name)."""
        logger = _caller_logger(stacklevel)
        kwargs = self._with_module(kwargs, module)
        if self._deferred_records is not None:
            self._deferred_records.append((logger, logging.INFO, message, args, kwargs))
            return
        logger.info(message, *args, stacklevel=stacklevel, **kwargs)

    def debug(
        self,
        message: object,
        *args: object,
        stacklevel: int = 2,
        module: Module | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log at DEBUG level on the caller's module logger."""
        logger = _caller_logger(stacklevel)
        kwargs = self._with_module(kwargs, module)
        if self._deferred_records is not None:
            self._deferred_records.append((logger, logging.DEBUG, message, args, kwargs))
            return
        logger.debug(message, *args, stacklevel=stacklevel, **kwargs)

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

    @contextmanager
    def deferred(self) -> Iterator[None]:
        """Capture warnings **and** INFO/DEBUG logs emitted inside the ``with``
        block, and replay them after it exits.

        Use this when an early, ordered block of console output (e.g. the
        startup summary panel) would otherwise be interleaved with messages
        raised during setup — render the panel inside the block, and the
        deferred messages replay after it. Warnings keep their original
        ``filename`` / ``lineno`` so the rich handler still locates the source;
        INFO/DEBUG keep their caller's module logger (captured at call time).
        Errors/criticals are never deferred — they surface immediately.
        """
        records: list[tuple[logging.Logger, int, object, tuple, dict]] = []
        previous_records = self._deferred_records
        self._deferred_records = records
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                yield
        finally:
            self._deferred_records = previous_records
        for entry in captured:
            warnings.warn_explicit(
                entry.message,
                entry.category,
                entry.filename,
                entry.lineno,
                source=entry.source,
            )
        for logger, level, message, args, kwargs in records:
            logger.log(level, message, *args, **kwargs)


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
