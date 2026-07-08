"""Tests for :mod:`raitap._config_schema` (JSON Schema generated from the
live adapter registry, refs #301)."""

from __future__ import annotations


def test_schema_lists_transparency_use_enum() -> None:
    import raitap.transparency  # noqa: F401 — fire discovery  # pyright: ignore[reportUnusedImport]
    from raitap._config_schema import build_config_schema

    schema = build_config_schema()
    enum = schema["properties"]["transparency"]["additionalProperties"]["properties"]["use"]["enum"]
    assert "captum" in enum and "shap" in enum
