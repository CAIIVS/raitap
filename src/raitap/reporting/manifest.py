from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sections import ReportGroup, ReportSection


@dataclass(frozen=True)
class ReportManifest:
    """Machine-readable description of a generated report."""

    kind: str
    sections: tuple[ReportSection, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    filename: str = "report.pdf"

    def write(self, path: Path, *, report_dir: Path) -> None:
        payload = {
            "kind": self.kind,
            "filename": self.filename,
            "metadata": self.metadata,
            "sections": [
                _section_to_dict(section, report_dir=report_dir) for section in self.sections
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ReportManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        report_dir = Path(path).parent
        sections = tuple(
            _section_from_dict(section_payload, report_dir=report_dir)
            for section_payload in payload.get("sections", [])
        )
        return cls(
            kind=str(payload.get("kind", "run")),
            sections=sections,
            metadata=dict(payload.get("metadata", {})),
            filename=str(payload.get("filename", "report.pdf")),
        )


def _section_to_dict(section: ReportSection, *, report_dir: Path) -> dict[str, Any]:
    return {
        "title": section.title,
        "metadata": section.metadata,
        "groups": [_group_to_dict(group, report_dir=report_dir) for group in section.groups],
    }


def _group_to_dict(group: ReportGroup, *, report_dir: Path) -> dict[str, Any]:
    return {
        "heading": group.heading,
        "table_rows": list(group.table_rows),
        "images": [_path_to_manifest_value(path, report_dir=report_dir) for path in group.images],
        "metadata": group.metadata,
    }


def _section_from_dict(payload: dict[str, Any], *, report_dir: Path) -> ReportSection:
    return ReportSection(
        title=str(payload.get("title", "")),
        groups=tuple(
            _group_from_dict(group_payload, report_dir=report_dir)
            for group_payload in payload.get("groups", [])
        ),
        metadata=dict(payload.get("metadata", {})),
    )


def _group_from_dict(payload: dict[str, Any], *, report_dir: Path) -> ReportGroup:
    return ReportGroup(
        heading=str(payload.get("heading", "")),
        table_rows=_table_rows_from_manifest(payload.get("table_rows", [])),
        images=tuple(
            _path_from_manifest_value(str(image), report_dir=report_dir)
            for image in payload.get("images", [])
        ),
        metadata=dict(payload.get("metadata", {})),
    )


def _table_rows_from_manifest(value: Any) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    if not isinstance(value, list):
        return ()

    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        name, cell_value = row
        rows.append((str(name), str(cell_value)))
    return tuple(rows)


def _path_to_manifest_value(path: Path, *, report_dir: Path) -> str:
    try:
        return str(path.relative_to(report_dir))
    except ValueError:
        return str(path)


def _path_from_manifest_value(value: str, *, report_dir: Path) -> Path:
    base_dir = report_dir.resolve()
    path = Path(value)
    candidate = path.resolve() if path.is_absolute() else (base_dir / path).resolve()
    try:
        candidate.relative_to(base_dir)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes report directory: {value}") from error
    return candidate
