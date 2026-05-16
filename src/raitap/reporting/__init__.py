"""Report generation module for RAITAP pipeline outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .builder import BuiltReport, build_merged_report, build_report
from .factory import ReportGeneration, create_report, reporting_enabled
from .html_reporter import HTMLReporter
from .manifest import ReportManifest
from .pdf_reporter import PDFReporter
from .sections import ReportGroup, ReportSection

if TYPE_CHECKING:
    from raitap.configs.schema import ReportingConfig


def __getattr__(name: str) -> Any:
    """Resolve hydra-zen builders by registry name, plus the schema dataclass
    (:class:`~raitap.configs.schema.ReportingConfig`) re-exported here."""
    if name == "ReportingConfig":
        from raitap.configs.schema import ReportingConfig

        return ReportingConfig
    from raitap._adapters import lookup

    return lookup("reporting", name)


__all__ = [
    "BuiltReport",
    "HTMLReporter",
    "PDFReporter",
    "ReportGeneration",
    "ReportGroup",
    "ReportManifest",
    "ReportSection",
    "ReportingConfig",
    "build_merged_report",
    "build_report",
    "create_report",
    "reporting_enabled",
]
