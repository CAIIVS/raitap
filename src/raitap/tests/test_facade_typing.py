"""Guards that the namespaced facades preserve decoration-site type checking."""

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


def _errors(src: str, tmp_path: Path) -> list[str]:
    f = tmp_path / "s.py"
    f.write_text(textwrap.dedent(src))
    assert pyright is not None
    proc = subprocess.run(
        [pyright, "--outputjson", str(f)], capture_output=True, text=True, check=False
    )
    data = json.loads(proc.stdout)
    return [d["message"] for d in data.get("generalDiagnostics", []) if d["severity"] == "error"]


def test_good_call_typechecks_clean(tmp_path: Path) -> None:
    errs = _errors(
        """
        import torch
        from raitap import adapters
        from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer

        @adapters.transparency(registry_name="x", algorithm_registry={})
        class _M(AttributionOnlyExplainer):
            def compute_attributions(self, model, inputs, **kwargs) -> torch.Tensor: ...
        """,
        tmp_path,
    )
    assert errs == [], errs


def test_missing_required_kwarg_errors(tmp_path: Path) -> None:
    errs = _errors(
        """
        from raitap import adapters

        @adapters.transparency(algorithm_registry={})
        class _M: ...
        """,
        tmp_path,
    )
    assert any("registry_name" in e for e in errs), errs
