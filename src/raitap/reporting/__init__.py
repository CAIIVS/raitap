"""Report generation module for RAITAP pipeline outputs."""

from .builder import BuiltReport, build_merged_report, build_report
from .factory import ReportGeneration, create_report, reporting_enabled
from .html_reporter import HTMLReporter
from .manifest import ReportManifest
from .pdf_reporter import PDFReporter
from .sections import ReportGroup, ReportSection


def __getattr__(name: str) -> object:
    """Resolve hydra-zen builders by registry name."""
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
    "build_merged_report",
    "build_report",
    "create_report",
    "reporting_enabled",
]
