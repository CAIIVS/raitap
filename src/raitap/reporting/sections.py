"""Generic structure for embedding module outputs (tables, figures) into reports."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReportGroup:
    """
    One headed block within a report section: optional scalar table and/or image files.

    ``table_rows`` is an ordered sequence of ``(name, value)`` string pairs rendered as a
    two-column table.  ``images`` is an ordered sequence of on-disk image paths rendered as
    figures.  A group may carry either, both, or neither.
    """

    heading: str
    images: tuple[Path, ...] = ()
    table_rows: tuple[tuple[str, str], ...] = ()


@runtime_checkable
class Reportable(Protocol):
    """
    Interface for objects that can contribute content to a report.
    """

    @abstractmethod
    def to_report_group(self) -> ReportGroup:
        """Return a ReportGroup representing this object's report content."""
        ...


@dataclass(frozen=True, slots=True)
class ReportSection:
    """
    One major PDF section with a title and ordered groups (e.g. Metrics, Transparency).
    """

    title: str
    groups: tuple[ReportGroup, ...]

    @classmethod
    def from_groups(cls, title: str, groups: Sequence[ReportGroup]) -> ReportSection:
        return cls(title=title, groups=tuple(groups))
