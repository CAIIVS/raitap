"""Tests for :mod:`raitap.utils.warnings`.

Covers warning-specific glue: :func:`suppress_warning` and the thread-local
diagnostic queue. Pure :class:`Diagnostic` data plumbing is tested in
``test_diagnostics.py``.
"""

from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING

import pytest

import raitap.utils.warnings as warnings_module
from raitap.utils.diagnostics import Diagnostic
from raitap.utils.warnings import suppress_warning

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clear_diagnostic_queue() -> Iterator[None]:
    warnings_module._clear_diagnostics()
    yield
    warnings_module._clear_diagnostics()


class TestSuppressWarning:
    def test_suppresses_matching_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            suppress_warning(message=r"silence-me", category=UserWarning)
            warnings.warn("silence-me please", UserWarning, stacklevel=1)
            warnings.warn("but-not-this", UserWarning, stacklevel=1)
        messages = [str(w.message) for w in caught]
        assert "silence-me please" not in messages
        assert "but-not-this" in messages

    def test_default_category_is_user_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            suppress_warning(message=r"hush")
            warnings.warn("hush now", UserWarning, stacklevel=1)
        assert all("hush now" not in str(w.message) for w in caught)


class TestDiagnosticQueueIsThreadLocal:
    def test_pushes_in_one_thread_dont_leak_into_another(self) -> None:
        from raitap.utils.warnings import _pop_diagnostic, _push_diagnostic

        sentinel = Diagnostic(subsystem="metrics", file="/x.py", line=1, third_party_lib=None)
        _push_diagnostic(sentinel)

        observed: list[Diagnostic | None] = []

        def _worker() -> None:
            observed.append(_pop_diagnostic())

        thread = threading.Thread(target=_worker)
        thread.start()
        thread.join()

        # Worker thread sees an empty queue (no leak from main thread).
        assert observed == [None]
        # Main-thread queue is still intact.
        assert _pop_diagnostic() == sentinel
