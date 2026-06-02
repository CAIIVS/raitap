"""Shared report-asset staging helpers.

Extracted from ``builder.py`` so per-phase report renderers
(``transparency/report.py``, ``robustness/report.py``) and the builder all
stage figures and copy assets through one implementation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from raitap.transparency.results import VisualisationResult


def _copy_asset(source: Path, *, assets_dir: Path, target_name: str) -> Path:
    target_name_path = Path(target_name)
    if target_name_path.is_absolute() or len(target_name_path.parts) != 1:
        raise ValueError(f"Asset target names must be simple filenames, got {target_name!r}.")

    target = assets_dir / target_name_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _strip_report_figure_titles(figure: Any) -> None:
    if hasattr(figure, "suptitle"):
        figure.suptitle("")
    for ax in getattr(figure, "axes", []):
        ax.set_title("")


def _stage_rendered_visualisations(
    visualisations: list[VisualisationResult],
    *,
    assets_dir: Path,
    file_stem_prefix: str,
    strip_titles: bool = False,
) -> tuple[Path, ...]:
    staged: list[Path] = []
    for visualisation in visualisations:
        target = assets_dir / f"{file_stem_prefix}_{visualisation.visualiser_name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if strip_titles:
            _strip_report_figure_titles(visualisation.figure)
        try:
            visualisation.figure.savefig(target, bbox_inches="tight", dpi=150)
        finally:
            plt.close(visualisation.figure)
        staged.append(target)
    return tuple(staged)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "asset"
