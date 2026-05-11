"""Error-rethrow layer for wrapped third-party libraries.

Sits on top of :mod:`raitap.utils.diagnostics`. Adapter call sites wrap
third-party library calls in :func:`rethrow`; when the library raises an
exception whose message matches a curated pattern, the original is replaced
with a user-actionable message and re-raised as :class:`AdapterError`, with
the original exception preserved on ``__cause__`` so the raw traceback stays
available for debugging.

Unmatched exceptions propagate untouched — the rethrow layer only rewords
known footguns, it never masks real bugs.

Companion to :mod:`raitap.utils.log` (warnings side of the same plumbing).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from raitap.utils.diagnostics import (
    Diagnostic,
    Subsystem,
    _classify_subsystem,
    _detect_third_party,
)

if TYPE_CHECKING:
    import re
    from collections.abc import Iterator, Mapping
    from types import TracebackType


class RaitapError(Exception):
    """Base raitap exception carrying a :class:`Diagnostic`.

    The diagnostic powers audience-aware rendering in
    :class:`raitap.utils.console.RaitapRichHandler` and
    :func:`raitap.utils.console.print_failure_panel`.
    """

    diagnostic: Diagnostic | None

    def __init__(self, message: str, *, diagnostic: Diagnostic | None = None) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


class AdapterError(RaitapError):
    """Raised by :func:`rethrow` when a wrapped library error is rewrapped.

    The original exception is preserved on ``__cause__``.
    """


def resolve_diagnostic_from_traceback(
    tb: TracebackType | None,
    *,
    default_file: str = "",
    default_line: int = 0,
) -> Diagnostic:
    """Walk a traceback's ``tb_next`` chain to classify the diagnostic origin.

    Mirrors :func:`raitap.utils.diagnostics.resolve_diagnostic_from_frames`
    but consumes a frozen traceback rather than the live call stack — needed
    because by the time the rethrow handler runs the exception has propagated
    and the offending frames are no longer on ``sys._getframe``.

    Returns a :class:`Diagnostic` with:

    - ``file`` / ``line``: deepest frame inside ``raitap/<subsystem>/`` that
      isn't ``raitap/utils/`` (so the user lands at the adapter, not the
      rethrow helper), falling back to ``default_file:default_line``.
    - ``subsystem``: the matching ``<subsystem>``, or ``None``.
    - ``third_party_lib``: name of a known wrapped library if any traceback
      frame lives inside it; otherwise ``None``.
    """
    rai_path: str | None = None
    rai_line: int = default_line
    rai_sub: Subsystem | None = None
    third_party: str | None = None

    cursor = tb
    while cursor is not None:
        frame = cursor.tb_frame
        path = frame.f_code.co_filename
        normalized = path.replace("\\", "/")
        if third_party is None:
            third_party = _detect_third_party(path)
        if "/raitap/" in normalized and "/raitap/utils/" not in normalized:
            sub = _classify_subsystem(path)
            if sub is not None:
                rai_path = path
                rai_line = cursor.tb_lineno
                rai_sub = sub
        cursor = cursor.tb_next

    if rai_path is None:
        return Diagnostic(
            subsystem=None,
            file=default_file,
            line=default_line,
            third_party_lib=third_party,
        )
    return Diagnostic(
        subsystem=rai_sub,
        file=rai_path,
        line=rai_line,
        third_party_lib=third_party,
    )


@contextmanager
def rethrow(
    *,
    subsystem: Subsystem,
    third_party_lib: str | None,
    message_map: Mapping[re.Pattern[str], str],
    base_exc: type[BaseException] = Exception,
) -> Iterator[None]:
    """Rewrap third-party errors matching a curated pattern dict.

    Adapter call sites wrap their third-party calls in::

        with rethrow(
            subsystem=Subsystem.transparency,
            third_party_lib="shap",
            message_map=type(self).error_messages,
        ):
            shap_values = explainer.shap_values(inputs)

    On a matching exception, raises :class:`AdapterError` carrying the
    replacement message and a :class:`Diagnostic` resolved from the traceback,
    with ``__cause__`` set to the original. Unmatched exceptions and anything
    outside ``base_exc`` (e.g. :class:`KeyboardInterrupt`) propagate as-is.
    """
    try:
        yield
    except base_exc as exc:
        original_message = str(exc)
        replacement: str | None = None
        for pattern, new_message in message_map.items():
            if pattern.search(original_message):
                replacement = new_message
                break
        if replacement is None:
            raise
        diagnostic = resolve_diagnostic_from_traceback(
            exc.__traceback__,
            default_file="",
            default_line=0,
        )
        if diagnostic.subsystem is None:
            diagnostic = Diagnostic(
                subsystem=subsystem,
                file=diagnostic.file,
                line=diagnostic.line,
                third_party_lib=diagnostic.third_party_lib or third_party_lib,
            )
        elif diagnostic.third_party_lib is None and third_party_lib is not None:
            diagnostic = Diagnostic(
                subsystem=diagnostic.subsystem,
                file=diagnostic.file,
                line=diagnostic.line,
                third_party_lib=third_party_lib,
            )
        raise AdapterError(replacement, diagnostic=diagnostic) from exc


__all__ = [
    "AdapterError",
    "RaitapError",
    "resolve_diagnostic_from_traceback",
    "rethrow",
]
