"""End-to-end inference checks against real example configs.

These tests guard against drift between the inference mapping and the hand-
maintained ``uv sync`` lines in ``examples/`` and ``scripts/``. They load the
YAML directly (no Hydra compose) so they remain runnable without a populated
artifacts directory.
"""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

from raitap.configs.extras.inference import infer_extras

REPO_ROOT = Path(__file__).resolve().parents[5]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_lwise_ham10000_assessment() -> None:
    cfg = _load(REPO_ROOT / "examples" / "lwise-ham10000" / "assessment.yaml")
    extras, _ = infer_extras(cfg, hardware="cpu")
    # README's `uv sync` line: torch-cpu, captum, metrics, reporting (html→jinja),
    # torchattacks, marabou (from marabou_linf block in YAML).
    assert {"torch-cpu", "captum", "metrics", "jinja", "torchattacks", "marabou"} <= extras


def test_lwise_ham10000_assessment_mlflow() -> None:
    # assessment_mlflow.yaml uses Hydra `defaults: [assessment]`; merge the base
    # in manually since this test bypasses Hydra compose.
    base = _load(REPO_ROOT / "examples" / "lwise-ham10000" / "assessment.yaml")
    overlay = _load(REPO_ROOT / "examples" / "lwise-ham10000" / "assessment_mlflow.yaml")
    cfg = {**base, **{k: v for k, v in overlay.items() if k != "defaults"}}
    extras, _ = infer_extras(cfg, hardware="cpu")
    assert {"torch-cpu", "captum", "metrics", "jinja", "torchattacks", "mlflow"} <= extras


def test_lwise_ham10000_with_xpu_picks_torch_intel() -> None:
    cfg = _load(REPO_ROOT / "examples" / "lwise-ham10000" / "assessment.yaml")
    extras, _ = infer_extras(cfg, hardware="xpu")
    assert "torch-intel" in extras
    assert "torch-cpu" not in extras
