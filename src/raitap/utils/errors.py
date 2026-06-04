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
    Module,
    _classify_module,
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


class ModelInputShapeError(RaitapError):
    """Raised when model inputs cannot be adapted to the expected shape.

    Two flavours:

    * Runtime mismatch — supplied tensor's per-sample numel disagrees with
      the expected input shape (set ``input_shape`` and ``expected_shape``).
    * Ambiguous declared shape — an ONNX graph declares two or more dynamic
      dims so no single reshape target exists (set ``expected_shape`` only).

    In both cases the message names ``data.input_metadata.shape`` as the
    override knob.
    """

    def __init__(
        self,
        *,
        expected_shape: tuple[int | None, ...],
        input_shape: tuple[int, ...] | None = None,
        diagnostic: Diagnostic | None = None,
    ) -> None:
        self.input_shape = input_shape
        self.expected_shape = expected_shape
        expected_display = tuple("N" if dim is None else dim for dim in expected_shape)
        if input_shape is None:
            message = (
                f"Model declares an ambiguous input shape {expected_display}: more than "
                "one dynamic dimension means no single reshape target exists. "
                "Set `data.input_metadata.shape` in your config to declare the non-batch "
                "input layout explicitly."
            )
        else:
            message = (
                f"Model input shape mismatch: got {input_shape}, expected "
                f"{expected_display}. Set `data.input_metadata.shape` in your config to "
                "declare the non-batch input layout (e.g. `[1, 1, 5]` for ACAS Xu), or "
                "check that the data loader emits the right rank."
            )
        super().__init__(message, diagnostic=diagnostic)


class SampleNamesLengthError(RaitapError):
    """Raised when ``raitap.sample_names`` length does not match batch size N.

    Surfaces at factory entry (``Explanation`` / ``RobustnessAssessment``) and
    as a defensive check in ``ExplanationResult`` / ``RobustnessResult``
    visualisation paths. ``None`` / empty lists are not errors (they mean
    "no labels"); only length mismatch raises.
    """

    def __init__(self, *, got: int, expected: int, source: str) -> None:
        self.got = got
        self.expected = expected
        message = (
            f"`raitap.sample_names` length mismatch: got {got} name(s), "
            f"expected {expected} (one per input sample). Source: {source}. "
            f"Either supply a list of length {expected} or omit "
            f"`sample_names` to use auto-derived sample ids."
        )
        super().__init__(message)


class BackendIncompatibilityError(Exception):
    """Raised when an adapter's algorithm needs capabilities the backend lacks.

    ``missing`` is the sorted list of capability values the backend does not
    provide (``algorithm.requires - backend.provides``).
    """

    def __init__(self, *, adapter: str, backend: str, missing: list[str]) -> None:
        self.adapter = adapter
        self.backend = backend
        self.missing = missing
        joined = ", ".join(missing) or "none"
        super().__init__(
            f"Adapter {adapter!r} is not compatible with backend {backend!r}.\n"
            f"Missing capabilities: {joined}. Use a backend that provides them "
            f"(a torch backend supplies autograd)."
        )


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

    - ``file`` / ``line``: deepest frame inside ``raitap/<module>/`` that
      isn't ``raitap/utils/`` (so the user lands at the adapter, not the
      rethrow helper), falling back to ``default_file:default_line``.
    - ``module``: the matching ``<module>``, or ``None``.
    - ``third_party_lib``: name of a known wrapped library if any traceback
      frame lives inside it; otherwise ``None``.
    """
    rai_path: str | None = None
    rai_line: int = default_line
    rai_sub: Module | None = None
    third_party: str | None = None

    cursor = tb
    while cursor is not None:
        frame = cursor.tb_frame
        path = frame.f_code.co_filename
        normalized = path.replace("\\", "/")
        if third_party is None:
            third_party = _detect_third_party(path)
        if "/raitap/" in normalized and "/raitap/utils/" not in normalized:
            sub = _classify_module(path)
            if sub is not None:
                rai_path = path
                rai_line = cursor.tb_lineno
                rai_sub = sub
        cursor = cursor.tb_next

    if rai_path is None:
        return Diagnostic(
            module=None,
            file=default_file,
            line=default_line,
            third_party_lib=third_party,
        )
    return Diagnostic(
        module=rai_sub,
        file=rai_path,
        line=rai_line,
        third_party_lib=third_party,
    )


@contextmanager
def rethrow(
    *,
    module: Module,
    third_party_lib: str | None,
    message_map: Mapping[re.Pattern[str], str],
    base_exc: type[BaseException] = Exception,
) -> Iterator[None]:
    """Rewrap third-party errors matching a curated pattern dict.

    Adapter call sites wrap their third-party calls in::

        with rethrow(
            module=Module.transparency,
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
        if diagnostic.module is None:
            diagnostic = Diagnostic(
                module=module,
                file=diagnostic.file,
                line=diagnostic.line,
                third_party_lib=diagnostic.third_party_lib or third_party_lib,
            )
        elif diagnostic.third_party_lib is None and third_party_lib is not None:
            diagnostic = Diagnostic(
                module=diagnostic.module,
                file=diagnostic.file,
                line=diagnostic.line,
                third_party_lib=third_party_lib,
            )
        raise AdapterError(replacement, diagnostic=diagnostic) from exc


__all__ = [
    "AdapterError",
    "BackendIncompatibilityError",
    "ModelInputShapeError",
    "RaitapError",
    "SampleNamesLengthError",
    "resolve_diagnostic_from_traceback",
    "rethrow",
]
