"""Generic structure for embedding module outputs (tables, figures) into reports."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from raitap.reporting.samples import SelectedSample


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
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Reportable(Protocol):
    """Interface for objects that contribute whole sections to a report.

    ``report_order`` positions this object's sections relative to other phases'
    (lower first); ``report_sections`` returns the ordered sections, staging any
    figures into ``ctx.assets_dir``.
    """

    report_order: ClassVar[int]

    @abstractmethod
    def report_sections(self, ctx: ReportContext) -> Sequence[ReportSection]:
        """Return this object's contribution as ordered report sections."""
        pass


@dataclass(frozen=True, slots=True)
class ReportSection:
    """
    One major PDF section with a title and ordered groups (e.g. Metrics, Transparency).
    """

    title: str
    groups: tuple[ReportGroup, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_groups(
        cls,
        title: str,
        groups: Sequence[ReportGroup],
        metadata: dict[str, object] | None = None,
    ) -> ReportSection:
        return cls(title=title, groups=tuple(groups), metadata={} if metadata is None else metadata)


@dataclass(frozen=True)
class ReportContext:
    """Cross-phase inputs a :class:`Reportable` needs to render its sections.

    ``selected_samples`` is computed once by the builder from the run's
    predictions (phase-agnostic); the rest are reporting-config flags.
    """

    assets_dir: Path
    selected_samples: tuple[SelectedSample, ...]
    show_original_per_explainer: bool = False
    show_redundant_robustness_panels: bool = False
    explicit_selection: bool = False
