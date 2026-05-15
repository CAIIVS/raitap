"""Adapter-level rethrow coverage for :class:`ShapExplainer`."""

from __future__ import annotations

import pytest

from raitap.transparency.explainers import ShapExplainer
from raitap.utils.diagnostics import Module
from raitap.utils.errors import AdapterError, rethrow


def test_error_messages_contains_silu_entry() -> None:
    """The seed pattern from issue #22 must match the original SHAP error text."""
    original = (
        "Output 0 of BackwardHookFunctionBackward is a view and is being modified "
        "inplace. This view was created inside a custom Function (or because an "
        "input was returned as-is) and the autograd logic to handle view+inplace "
        "would override the custom backward associated with the custom Function, "
        "leading to incorrect gradients. This behavior is forbidden."
    )
    patterns = ShapExplainer.error_patterns
    assert patterns is not None
    hits = [replacement for pattern, replacement in patterns.items() if pattern.search(original)]
    assert hits, "Expected the SiLU/DeepExplainer pattern to match the issue's example error."
    assert "GradientExplainer" in hits[0]


def test_silu_runtime_error_is_rewrapped() -> None:
    """End-to-end: rethrow context using ShapExplainer.error_patterns rewraps the
    canonical SiLU :class:`RuntimeError` into an :class:`AdapterError`.

    Exercising the dict via :func:`rethrow` directly avoids needing the optional
    ``shap`` dependency, while still proving the dict + wrapper compose correctly.
    """
    patterns = ShapExplainer.error_patterns
    assert patterns is not None
    with (
        pytest.raises(AdapterError) as info,
        rethrow(
            module=Module.transparency,
            third_party_lib="shap",
            message_map=patterns,
        ),
    ):
        raise RuntimeError(
            "Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace."
        )
    exc = info.value
    assert "GradientExplainer" in str(exc)
    assert isinstance(exc.__cause__, RuntimeError)
    assert exc.diagnostic is not None
    assert exc.diagnostic.module == Module.transparency
    assert exc.diagnostic.third_party_lib == "shap"
