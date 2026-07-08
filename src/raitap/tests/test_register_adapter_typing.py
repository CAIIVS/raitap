"""Static-type assertions: pyright must error at the decoration site when an
adapter author forgets the ``registry_name`` cross-family required kwarg.

The whole point of ``AdapterDecoratorOptions.registry_name: Required[str]`` + PEP 692
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
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

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
    _tr_call = "adapters.transparency(algorithm_registry={}, extra='x', import_name='x')"
    _ro_call = "adapters.robustness(algorithm_registry={}, extra='x', import_name='x')"
    for decorator_import, decorator_call, expected in [
        ("from raitap import adapters", _tr_call, "registry_name"),
        ("from raitap import adapters", _ro_call, "registry_name"),
        ("from raitap import adapters", "adapters.metrics(extra='x')", "registry_name"),
        ("from raitap import adapters", "adapters.reporter(extra='x')", "registry_name"),
        ("from raitap import adapters", "adapters.tracker(extra='x')", "registry_name"),
        ("from raitap import visualisers", "visualisers.transparency()", "registry_name"),
        ("from raitap import visualisers", "visualisers.robustness()", "registry_name"),
        ("from raitap import backends", "backends.register()", "provides"),
    ]:
        errors = _pyright_errors(
            f"""
            {decorator_import}

            @{decorator_call}
            class _M: ...
            """,
            tmp_path,
        )
        assert any(expected in e for e in errors), (
            f"{decorator_call} did not produce a {expected!r} pyright error. Got: {errors}"
        )
