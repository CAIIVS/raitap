"""Generic structure for embedding module figure outputs (e.g. PNGs) into reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReportImageGroup:
    """Named set of raster figures on disk (typically Matplotlib-exported PNGs)."""

    heading: str
    run_dir: Path
    glob_pattern: str = "*.png"


@dataclass(frozen=True, slots=True)
class ReportImageSection:
    """
    One major PDF section with a title and ordered figure groups (e.g. Transparency, Robustness).
    """

    title: str
    groups: tuple[ReportImageGroup, ...]

    @classmethod
    def from_groups(cls, title: str, groups: Sequence[ReportImageGroup]) -> ReportImageSection:
        return cls(title=title, groups=tuple(groups))
