"""Shared test factories and helpers (importable from any test module)."""

from __future__ import annotations

from raitap.testing.deps import requires
from raitap.testing.factories import make_app_config, make_tiny_classifier, make_tiny_mlp

__all__ = ["make_app_config", "make_tiny_classifier", "make_tiny_mlp", "requires"]
