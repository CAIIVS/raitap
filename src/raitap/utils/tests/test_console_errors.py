"""Console rendering of :class:`RaitapError` panels.

Focuses on the chip-composition logic and the :func:`print_failure_panel`
``RaitapError`` branch — both newly added for issue #22 to bring error panels
to parity with the warning rendering shipped in PR #124.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from rich.console import Console

import raitap.utils.console as console_module
from raitap.utils.colour import THEME, Status
from raitap.utils.console import (
    RaitapRichHandler,
    diagnostic_chips,
    print_failure_panel,
)
from raitap.utils.diagnostics import Diagnostic, Subsystem
from raitap.utils.errors import AdapterError, RaitapError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture(autouse=True)
def _reset_dev_install_cache() -> Iterator[None]:
    from raitap.utils.diagnostics import is_dev_install

    is_dev_install.cache_clear()
    yield
    is_dev_install.cache_clear()


class TestDiagnosticChips:
    def test_dev_install_includes_path_and_third_party(self) -> None:
        diag = Diagnostic(
            subsystem=Subsystem.transparency,
            file="src/raitap/transparency/explainers/shap_explainer.py",
            line=196,
            third_party_lib="shap",
        )
        with patch.object(console_module, "is_dev_install", return_value=True):
            chips = diagnostic_chips(
                Status.ERROR,
                scope="Transparency",
                src=f"{diag.file}:{diag.line}",
                diagnostic=diag,
            )
        joined = " ".join(c.plain for c in chips)
        assert "Transparency" in joined
        assert "shap_explainer.py:196" in joined
        assert "via Shap" in joined

    def test_installed_wheel_includes_docs_link(self) -> None:
        diag = Diagnostic(
            subsystem=Subsystem.transparency,
            file="<frozen>",
            line=0,
            third_party_lib="shap",
        )
        with patch.object(console_module, "is_dev_install", return_value=False):
            chips = diagnostic_chips(
                Status.ERROR,
                scope="Transparency",
                src="",
                diagnostic=diag,
            )
        joined = " ".join(c.plain for c in chips)
        assert "View docs" in joined
        assert "<frozen>" not in joined

    def test_unclassified_in_installed_mode_yields_no_chips(self) -> None:
        diag = Diagnostic(subsystem=None, file="", line=0, third_party_lib=None)
        with patch.object(console_module, "is_dev_install", return_value=False):
            chips = diagnostic_chips(
                Status.ERROR,
                scope="RaitapError",
                src="",
                diagnostic=diag,
            )
        assert chips == []


class TestPrintFailurePanel:
    def test_plain_exception_uses_default_title(self) -> None:
        console = Console(file=io.StringIO(), force_terminal=False, width=120, theme=THEME)
        with patch.object(console_module, "get_stderr_console", return_value=console):
            print_failure_panel(RuntimeError("plain boom"), "0:00:01")
        output = console.file.getvalue()  # type: ignore[attr-defined]
        assert "Assessment failed" in output
        assert "RuntimeError" in output
        assert "plain boom" in output

    def test_raitap_error_renders_diagnostic_chips_and_cause(self) -> None:
        diag = Diagnostic(
            subsystem=Subsystem.transparency,
            file="/x/raitap/transparency/explainers/shap_explainer.py",
            line=196,
            third_party_lib="shap",
        )
        try:
            try:
                raise RuntimeError("original library error")
            except RuntimeError as exc:
                raise AdapterError("user-facing replacement", diagnostic=diag) from exc
        except AdapterError as exc:
            captured = exc

        console = Console(file=io.StringIO(), force_terminal=False, width=160, theme=THEME)
        with (
            patch.object(console_module, "get_stderr_console", return_value=console),
            patch.object(console_module, "is_dev_install", return_value=True),
        ):
            print_failure_panel(captured, "0:00:01")
        output = console.file.getvalue()  # type: ignore[attr-defined]
        assert "user-facing replacement" in output
        assert "Transparency" in output
        assert "caused by RuntimeError" in output


def _emit_to_handler(record_factory: Callable[[logging.Logger], None]) -> str:
    console = Console(file=io.StringIO(), force_terminal=False, width=160, theme=THEME)
    handler = RaitapRichHandler(console=console, show_time=False, show_level=False)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("raitap.tests.console_fallback")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        record_factory(logger)
    finally:
        logger.removeHandler(handler)
    return console.file.getvalue()  # type: ignore[attr-defined]


class TestRichHandlerErrorPanel:
    def test_plain_logger_exception_renders_subsystem_chip_from_traceback(self) -> None:
        # Simulate an exception caught inside src/raitap/robustness/ so the
        # traceback walker classifies the frame as the robustness subsystem.
        src = "import logging\ndef boom():\n    raise RuntimeError('inner')\n"
        code = compile(src, "/tmp/raitap/src/raitap/robustness/fake.py", "exec")
        ns: dict[str, object] = {}
        exec(code, ns)

        def fire(logger: logging.Logger) -> None:
            with patch.object(console_module, "is_dev_install", return_value=True):
                try:
                    ns["boom"]()  # type: ignore[operator]
                except RuntimeError:
                    logger.exception("verify_sample crashed for index 0")

        output = _emit_to_handler(fire)
        assert "verify_sample crashed" in output
        assert "Robustness" in output
        assert "fake.py:3" in output

    def test_plain_logger_warning_falls_back_to_record_path(self) -> None:
        def fire(logger: logging.Logger) -> None:
            with patch.object(console_module, "is_dev_install", return_value=True):
                # Synthesise a record whose pathname lives in a raitap subsystem
                # but whose message does NOT match the header pattern.
                record = logger.makeRecord(
                    name=logger.name,
                    level=logging.WARNING,
                    fn="/tmp/raitap/src/raitap/metrics/factory.py",
                    lno=51,
                    msg="bare warning with no header",
                    args=(),
                    exc_info=None,
                )
                logger.handle(record)

        output = _emit_to_handler(fire)
        assert "Metrics" in output
        assert "factory.py:51" in output

    def test_logger_error_with_raitap_exc_renders_diagnostic_header(self) -> None:
        diag = Diagnostic(
            subsystem=Subsystem.robustness,
            file="/x/raitap/robustness/assessors/foolbox_assessor.py",
            line=161,
            third_party_lib="foolbox",
        )
        err = RaitapError("rewrapped robustness failure", diagnostic=diag)

        console = Console(file=io.StringIO(), force_terminal=False, width=160, theme=THEME)
        handler = RaitapRichHandler(console=console, show_time=False, show_level=False)
        handler.setLevel(logging.ERROR)

        logger = logging.getLogger("raitap.tests.console_errors")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        logger.propagate = False

        try:
            with patch.object(console_module, "is_dev_install", return_value=True):
                try:
                    raise err
                except RaitapError:
                    logger.error("rewrapped robustness failure", exc_info=True)
        finally:
            logger.removeHandler(handler)

        output = console.file.getvalue()  # type: ignore[attr-defined]
        assert "Robustness" in output
        assert "via Foolbox" in output
