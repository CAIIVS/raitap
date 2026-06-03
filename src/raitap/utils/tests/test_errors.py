"""Tests for :mod:`raitap.utils.errors`."""

from __future__ import annotations

import re
import types

import pytest

from raitap.utils.diagnostics import Diagnostic, Module
from raitap.utils.errors import (
    AdapterError,
    RaitapError,
    resolve_diagnostic_from_traceback,
    rethrow,
)


class TestResolveDiagnosticFromTraceback:
    def test_returns_default_when_tb_is_none(self) -> None:
        diag = resolve_diagnostic_from_traceback(None, default_file="x.py", default_line=7)
        assert diag.file == "x.py"
        assert diag.line == 7
        assert diag.module is None
        assert diag.third_party_lib is None

    def test_picks_deepest_raitap_module_frame(self) -> None:
        # Build a synthetic traceback chain: outer non-raitap → transparency → utils.
        tb_inner_utils = _make_tb("/x/raitap/utils/log.py", 20, None)
        tb_transparency = _make_tb(
            "/x/raitap/transparency/explainers/shap_explainer.py", 196, tb_inner_utils
        )
        tb_root = _make_tb("/other/site.py", 1, tb_transparency)
        diag = resolve_diagnostic_from_traceback(tb_root)  # pyright: ignore[reportArgumentType]
        assert diag.module == Module.transparency
        assert diag.line == 196
        assert "shap_explainer.py" in diag.file

    def test_detects_third_party_in_chain(self) -> None:
        tb_inner = _make_tb("/x/site-packages/shap/explainers/_deep.py", 42, None)
        tb_outer = _make_tb("/x/raitap/transparency/explainers/shap_explainer.py", 196, tb_inner)
        diag = resolve_diagnostic_from_traceback(tb_outer)  # pyright: ignore[reportArgumentType]
        assert diag.third_party_lib == "shap"
        assert diag.module == Module.transparency


class TestRethrow:
    def test_passthrough_when_no_exception(self) -> None:
        with rethrow(
            module=Module.transparency,
            third_party_lib="shap",
            message_map={re.compile("never matches"): "replacement"},
        ):
            value = 1 + 1
        assert value == 2

    def test_matches_pattern_and_wraps(self) -> None:
        message_map = {re.compile(r"view and is being modified inplace"): "Use GradientExplainer."}
        with (
            pytest.raises(AdapterError) as info,
            rethrow(
                module=Module.transparency,
                third_party_lib="shap",
                message_map=message_map,
            ),
        ):
            raise RuntimeError(
                "Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace"
            )
        exc = info.value
        assert "Use GradientExplainer" in str(exc)
        assert isinstance(exc.__cause__, RuntimeError)
        assert exc.diagnostic is not None
        assert exc.diagnostic.module == Module.transparency
        assert exc.diagnostic.third_party_lib == "shap"

    def test_unmatched_message_propagates_unchanged(self) -> None:
        with (
            pytest.raises(RuntimeError) as info,
            rethrow(
                module=Module.transparency,
                third_party_lib="shap",
                message_map={re.compile("xyz"): "rewritten"},
            ),
        ):
            raise RuntimeError("unrelated boom")
        assert not isinstance(info.value, AdapterError)
        assert str(info.value) == "unrelated boom"

    def test_skips_exceptions_outside_base_exc(self) -> None:
        with (
            pytest.raises(KeyboardInterrupt),
            rethrow(
                module=Module.transparency,
                third_party_lib="shap",
                message_map={re.compile(".*"): "rewritten"},
                base_exc=Exception,
            ),
        ):
            raise KeyboardInterrupt()


class TestRaitapError:
    def test_carries_diagnostic(self) -> None:
        diag = Diagnostic(
            module=Module.robustness,
            file="x.py",
            line=1,
            third_party_lib="foolbox",
        )
        err = RaitapError("boom", diagnostic=diag)
        assert err.diagnostic is diag
        assert str(err) == "boom"


def test_backend_incompatibility_message_lists_missing() -> None:
    from raitap.utils.errors import BackendIncompatibilityError

    err = BackendIncompatibilityError(adapter="PGD", backend="OnnxBackend", missing=["autograd"])
    assert err.adapter == "PGD"
    assert err.backend == "OnnxBackend"
    assert err.missing == ["autograd"]
    text = str(err)
    assert "PGD" in text and "OnnxBackend" in text and "autograd" in text


def _make_tb(filename: str, lineno: int, tb_next: object) -> object:
    """Build a duck-typed traceback object good enough for our walker.

    Real ``types.TracebackType`` cannot be instantiated directly; the walker
    only reads ``tb_frame.f_code.co_filename``, ``tb_lineno``, and ``tb_next``.
    """
    code = types.SimpleNamespace(co_filename=filename)
    frame = types.SimpleNamespace(f_code=code)
    return types.SimpleNamespace(tb_frame=frame, tb_lineno=lineno, tb_next=tb_next)
