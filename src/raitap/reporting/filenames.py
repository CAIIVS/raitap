"""Report output filename helpers."""

from __future__ import annotations

from pathlib import Path


def report_output_filename(filename: str, extension: str) -> str:
    """Return a simple report filename with the reporter-specific extension."""
    raw = str(filename).strip()
    if not raw:
        raise ValueError("reporting.filename must be a simple filename basename.")
    if Path(raw).name != raw or "/" in raw or "\\" in raw:
        raise ValueError(f"reporting.filename must be a simple filename, got {filename!r}.")

    suffix = extension if extension.startswith(".") else f".{extension}"
    basename = Path(raw).with_suffix("").name
    if not basename:
        raise ValueError("reporting.filename must be a simple filename basename.")
    return f"{basename}{suffix}"
