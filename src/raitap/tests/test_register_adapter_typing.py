"""Static-type assertions: pyright must error at the decoration site when an
adapter author forgets the ``registry_name`` cross-family required kwarg.

The whole point of ``_CommonRegKwargs.registry_name: Required[str]`` + PEP 692
``Unpack[...]`` on every family decorator is exactly this guarantee — verify
it empirically so a future drift in the typing surface fails CI loudly.

family-specific class-body attrs (``algorithm_registry``, ``output_payload_kind``)
are enforced at runtime by ``_register_core`` via ``FamilyConfig`` flags —
runtime coverage tested elsewhere.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

pyright = shutil.which("pyright")
pytestmark = pytest.mark.skipif(pyright is None, reason="pyright not on PATH")


def _pyright_errors(source: str, tmp_path: Path) -> list[str]:
    """Run pyright on ``source`` and return error-severity diagnostic messages."""
    f = tmp_path / "snippet.py"
    f.write_text(textwrap.dedent(source))
    assert pyright is not None  # narrowed by pytestmark — keeps type checkers happy
    proc = subprocess.run(
        [pyright, "--outputjson", str(f)],
        capture_output=True,
        text=True,
        check=False,
    )
    data = json.loads(proc.stdout)
    return [d["message"] for d in data.get("generalDiagnostics", []) if d["severity"] == "error"]


def test_missing_registry_name_is_pyright_error_for_every_family_decorator(
    tmp_path: Path,
) -> None:
    for decorator_import, decorator_call in [
        (
            "from raitap.transparency.explainers.registration import register_transparency_adapter",
            "register_transparency_adapter(extra='x', library='x')",
        ),
        (
            "from raitap.robustness.assessors.registration import register_robustness_adapter",
            "register_robustness_adapter(extra='x', library='x')",
        ),
        (
            "from raitap.metrics.registration import register_metrics_adapter",
            "register_metrics_adapter(extra='x')",
        ),
        (
            "from raitap.reporting.registration import register_reporter",
            "register_reporter(extra='x')",
        ),
        (
            "from raitap.tracking.registration import register_tracker",
            "register_tracker(extra='x')",
        ),
        (
            "from raitap.transparency.visualisers.registration import "
            "register_transparency_visualiser",
            "register_transparency_visualiser()",
        ),
        (
            "from raitap.robustness.visualisers.registration import "
            "register_robustness_visualiser",
            "register_robustness_visualiser()",
        ),
    ]:
        errors = _pyright_errors(
            f"""
            {decorator_import}

            @{decorator_call}
            class _M: ...
            """,
            tmp_path,
        )
        assert any("registry_name" in e for e in errors), (
            f"{decorator_call} did not produce a registry_name pyright error. "
            f"Got: {errors}"
        )
