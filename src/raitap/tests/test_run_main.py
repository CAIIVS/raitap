from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch

    from raitap.configs.schema import AppConfig

from raitap import run as run_module


def test_hydra_main_composes_default_config(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(config: object) -> None:
        captured["config"] = config

    monkeypatch.setattr(run_module, "run", fake_run)
    monkeypatch.setattr(run_module, "print_summary", lambda _cfg: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "raitap.run",
            f"hydra.run.dir={tmp_path / 'hydra-run'}",
            "hydra.output_subdir=null",
        ],
    )

    run_module.main()

    cfg = cast("AppConfig", captured["config"])
    assert cfg.model.source == "vit_b_32"
    assert cfg.metrics._target_ == "ClassificationMetrics"
    assert cfg.transparency
