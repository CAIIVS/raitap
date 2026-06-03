"""Tests for :mod:`raitap.utils.log`.

Covers warning-specific glue: :meth:`raitap_log.suppress` and the thread-local
diagnostic queue. Pure :class:`Diagnostic` data plumbing is tested in
``test_diagnostics.py``.
"""

from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING

import pytest

import raitap.utils.log as log_module
from raitap import raitap_log
from raitap.utils.diagnostics import Diagnostic, Module

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clear_diagnostic_queue() -> Iterator[None]:
    log_module._clear_diagnostics()
    yield
    log_module._clear_diagnostics()


class TestSuppress:
    def test_suppresses_matching_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raitap_log.suppress(message=r"silence-me", category=UserWarning)
            warnings.warn("silence-me please", UserWarning, stacklevel=1)
            warnings.warn("but-not-this", UserWarning, stacklevel=1)
        messages = [str(w.message) for w in caught]
        assert "silence-me please" not in messages
        assert "but-not-this" in messages

    def test_default_category_is_user_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raitap_log.suppress(message=r"hush")
            warnings.warn("hush now", UserWarning, stacklevel=1)
        assert all("hush now" not in str(w.message) for w in caught)


class TestStacklevelAttribution:
    def test_warn_blames_immediate_caller(self) -> None:
        """``raitap_log.warn`` default stacklevel must point ``filename`` at the
        caller, not at the facade body inside ``log.py``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raitap_log.warn("stacklevel-probe")
        assert len(caught) == 1
        assert caught[0].filename == __file__

    def test_info_blames_immediate_caller(self, caplog: pytest.LogCaptureFixture) -> None:
        """``raitap_log.info`` default stacklevel must point ``pathname`` at the
        caller, not at the facade body inside ``log.py``."""
        caplog.set_level("INFO")
        raitap_log.info("stacklevel-probe")
        records = [r for r in caplog.records if r.message == "stacklevel-probe"]
        assert len(records) == 1
        assert records[0].pathname == __file__


class TestDeferred:
    def test_info_is_buffered_during_block_and_replayed_after(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """INFO inside ``deferred()`` must not emit during the block (so an
        ordered panel can print first) and must replay on exit, keeping the
        caller's module logger."""
        caplog.set_level("INFO")
        with raitap_log.deferred():
            raitap_log.info("deferred-line")
            assert not [r for r in caplog.records if r.message == "deferred-line"]
        replayed = [r for r in caplog.records if r.message == "deferred-line"]
        assert len(replayed) == 1
        assert replayed[0].name == __name__

    def test_error_is_not_deferred(self, caplog: pytest.LogCaptureFixture) -> None:
        """Errors must surface immediately even inside a deferred block."""
        caplog.set_level("ERROR")
        with raitap_log.deferred():
            raitap_log.error("immediate-error")
            assert [r for r in caplog.records if r.message == "immediate-error"]


class TestDiagnosticQueueIsThreadLocal:
    def test_pushes_in_one_thread_dont_leak_into_another(self) -> None:
        from raitap.utils.log import _pop_diagnostic, _push_diagnostic

        sentinel = Diagnostic(module=Module.metrics, file="/x.py", line=1, third_party_lib=None)
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
