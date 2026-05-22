"""Canonical optional-dependency gating for tests.

Replaces the three pre-existing styles (module-level ``pytest.importorskip``,
``needs_*`` fixtures, inline ``importorskip``) with one decorator.
"""

from __future__ import annotations

import importlib.util

import pytest


def requires(*modules: str) -> pytest.MarkDecorator:
    """Skip the decorated test if any named import is unavailable.

    Usage::

        @requires("foolbox")
        def test_x(): ...
    """
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    reason = f"requires missing module(s): {', '.join(missing)}" if missing else ""
    return pytest.mark.skipif(bool(missing), reason=reason)
