"""Warning-specific glue on top of :mod:`raitap.utils.diagnostics`.

Adapters use :func:`suppress_warning` to silence noisy library warnings at
import time. The thread-local diagnostic queue is the bridge from
:func:`warnings.formatwarning` (which sees frames at warn time) to the rich
log handler (which formats panels at emit time, after frames are unwound).

Errors will get their own module (issue #22) that consumes the same
:class:`~raitap.utils.diagnostics.Diagnostic` from a traceback rather than
from a live frame walk.
"""

from __future__ import annotations

import sys
import threading
import warnings
from collections import deque

from raitap.utils.diagnostics import Diagnostic

# Per-thread FIFO of diagnostics resolved at ``warnings.formatwarning`` time,
# drained by the rich handler when it parses a warning panel. Thread-local so
# concurrent ``warnings.warn`` calls (e.g. dataloader workers) can't shuffle
# diagnostics between unrelated panels. Emit order matches warn order *within*
# a thread, which is what the warnings/logging pipeline already guarantees.
_diagnostic_state = threading.local()


def _diagnostic_queue() -> deque[Diagnostic]:
    queue = getattr(_diagnostic_state, "queue", None)
    if queue is None:
        queue = deque()
        _diagnostic_state.queue = queue
    return queue


def suppress_warning(
    *,
    message: str,
    category: type[Warning] = UserWarning,
    module: str = "",
) -> None:
    """Register a runtime ``warnings.filterwarnings("ignore", ...)`` filter.

    Adapters call this at import time to silence known-noise warnings their
    wrapped library emits. The match arguments mirror :func:`warnings.filterwarnings`.
    """
    warnings.filterwarnings("ignore", message=message, category=category, module=module)


def raitap_warn(
    message: str,
    *,
    subsystem: str | None = None,
    third_party_lib: str | None = None,
    category: type[Warning] = UserWarning,
    stacklevel: int = 2,
) -> None:
    """Emit a :class:`UserWarning` with explicit raitap subsystem metadata.

    Use this from call sites where the *logical* subsystem differs from the
    file path the warning is emitted from. For example, a robustness-related
    warning raised inside ``run/pipeline.py`` should still render under the
    ``Robustness`` chip with a link to the robustness docs page.

    When ``subsystem`` is omitted this behaves exactly like ``warnings.warn``;
    the rich handler will fall back to walking frames to classify the origin.
    """
    if subsystem is not None:
        frame = sys._getframe(max(stacklevel - 1, 0))
        _diagnostic_override.value = Diagnostic(
            subsystem=subsystem,
            file=frame.f_code.co_filename,
            line=frame.f_lineno,
            third_party_lib=third_party_lib,
        )
    try:
        warnings.warn(message, category, stacklevel=stacklevel + 1)
    finally:
        _diagnostic_override.value = None


# Thread-local override consulted by :func:`_format_warning_compact` before it
# falls back to a frame walk. Set by :func:`raitap_warn` and cleared in its
# ``finally`` block; ``None`` between calls.
_diagnostic_override = threading.local()
_diagnostic_override.value = None


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


__all__ = ["raitap_warn", "suppress_warning"]
