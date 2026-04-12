"""Report generation module for RAITAP pipeline outputs."""

from __future__ import annotations

from typing import Any

from .factory import ReportGeneration, create_report, reporting_enabled

__all__ = ["PDFReporter", "ReportGeneration", "create_report", "reporting_enabled"]


def __getattr__(name: str) -> Any:
    """Lazy ``PDFReporter`` so ``borb`` (optional ``reporting`` extra) is not imported at collection time."""
    if name == "PDFReporter":
        from .pdf_reporter import PDFReporter

        return PDFReporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
