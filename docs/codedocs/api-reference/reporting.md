---
title: "Reporting API"
description: "Reference for report building, report generation, and report dataclasses."
---

The reporting package turns `RunOutputs` into a PDF plus a machine-readable manifest. Its public surface includes both structural dataclasses and factory helpers.

## Imports

```python
from raitap.reporting import (
    BuiltReport,
    PDFReporter,
    ReportGeneration,
    ReportGroup,
    ReportManifest,
    ReportSection,
    build_merged_report,
    build_report,
    create_report,
    reporting_enabled,
)
```

## Factory helpers

### `reporting_enabled`

```python
def reporting_enabled(config: AppConfig) -> bool
```

Returns `True` when `config.reporting._target_` is a non-empty string.

### `build_report`

Source: `src/raitap/reporting/builder.py`

```python
def build_report(config: AppConfig, outputs: RunOutputs) -> BuiltReport
```

Creates a single-run report model with metrics, global, cohort, and local sections when those artifacts exist.

### `build_merged_report`

```python
def build_merged_report(
    config: AppConfig,
    *,
    sweep_dir: Path,
    child_manifests: list[tuple[str, str | None, ReportManifest]],
    skipped_children: list[str],
) -> BuiltReport
```

Builds a multirun report by merging previously written manifests.

### `create_report`

```python
def create_report(config: AppConfig, report: BuiltReport) -> ReportGeneration
```

Resolves the configured reporter, generates the PDF, writes `report_manifest.json`, and returns a `ReportGeneration`.

## Structural types

### `BuiltReport`

```python
@dataclass(frozen=True)
class BuiltReport:
    report_dir: Path
    sections: tuple[ReportSection, ...]
    manifest: ReportManifest
```

### `ReportGeneration`

```python
@dataclass
class ReportGeneration(Trackable):
    report_path: Path
    reporter: BaseReporter
    manifest_path: Path
```

Public method:

```python
def log(self, tracker: BaseTracker | None, **kwargs: Any) -> None
```

### `ReportGroup`

```python
@dataclass(frozen=True slots=True)
class ReportGroup:
    heading: str
    images: tuple[Path, ...] = ()
    table_rows: tuple[tuple[str, str], ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
```

### `ReportSection`

```python
@dataclass(frozen=True slots=True)
class ReportSection:
    title: str
    groups: tuple[ReportGroup, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_groups(
        cls,
        title: str,
        groups: Sequence[ReportGroup],
        metadata: dict[str, object] | None = None,
    ) -> ReportSection
```

### `ReportManifest`

```python
@dataclass(frozen=True)
class ReportManifest:
    kind: str
    sections: tuple[ReportSection, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    filename: str = "report.pdf"

    def write(self, path: Path, *, report_dir: Path) -> None
    @classmethod
    def load(cls, path: Path) -> ReportManifest
```

## Reporter implementation

### `PDFReporter`

Source: `src/raitap/reporting/pdf_reporter.py`

```python
class PDFReporter(BaseReporter):
    def generate(
        self,
        sections: Sequence[ReportSection],
        *,
        report_dir: Path | None = None,
    ) -> Path
```

`PDFReporter` uses the optional `borb` dependency to render the report and honors `reporting.formatting.*` values from the config.

## Common pattern

```python
from raitap.reporting import build_report, create_report

report = build_report(cfg, outputs)
generation = create_report(cfg, report)
print(generation.report_path)
```
