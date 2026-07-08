"""Tests for :mod:`raitap._config_schema` (JSON Schema generated from the
live adapter registry, refs #301)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_schema_lists_transparency_use_enum() -> None:
    import raitap.transparency  # noqa: F401 — fire discovery  # pyright: ignore[reportUnusedImport]
    from raitap._config_schema import build_config_schema

    schema = build_config_schema()
    enum = schema["properties"]["transparency"]["additionalProperties"]["properties"]["use"]["enum"]
    assert "captum" in enum and "shap" in enum


def test_committed_schema_matches_fresh_regen(tmp_path: Path) -> None:
    """The committed ``src/raitap/schema/raitap.schema.json`` must byte-match a
    fresh ``raitap config-schema`` regen (this is CI's freshness check, run
    locally too).

    Regenerates in a subprocess — a fresh interpreter, like a real ``raitap
    config-schema`` invocation — rather than calling :func:`build_config_schema`
    in-process. The family-registration import order is process-global
    (:mod:`raitap._adapters`'s ``_BUILDERS``), so whatever another test in this
    same pytest session already imported would otherwise leak in and make this
    comparison depend on test order.
    """
    committed_path = Path(__file__).resolve().parents[1] / "schema" / "raitap.schema.json"
    output_path = tmp_path / "raitap.schema.json"

    subprocess.run(
        [sys.executable, "-m", "raitap.cli", "config-schema", "-o", str(output_path)],
        check=True,
    )

    assert committed_path.read_text() == output_path.read_text(), (
        f"{committed_path} is stale — regenerate with "
        "`uv run raitap config-schema -o src/raitap/schema/raitap.schema.json`"
    )
