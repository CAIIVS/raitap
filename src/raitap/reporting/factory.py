from __future__ import annotations

import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hydra.utils import instantiate

from raitap import raitap_log
from raitap.configs import cfg_to_dict, resolve_target
from raitap.tracking.base_tracker import BaseTracker, Trackable

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

    from .base_reporter import BaseReporter
    from .builder import BuiltReport


_REPORTING_PREFIX = "raitap.reporting."


def reporting_enabled(config: AppConfig) -> bool:
    """Check if reporting is enabled in config."""
    reporting_cfg = config.reporting
    if reporting_cfg is None:
        return False
    target = getattr(reporting_cfg, "_target_", None)
    if target is None:
        return False
    return bool(str(target).strip())


@dataclass
class ReportGeneration(Trackable):
    """Outcome of report generation."""

    report_path: Path
    reporter: BaseReporter
    manifest_path: Path
    archive_path: Path | None = None

    def log(self, tracker: BaseTracker | None, **kwargs: Any) -> None:
        """Upload report to tracking system if configured."""
        if tracker is None:
            return

        # Forward entire report file as artifact
        tracker.log_artifacts(
            source_directory=self.report_path.parent,
            target_subdirectory="reports",
        )


def create_report(
    config: AppConfig,
    report: BuiltReport,
) -> ReportGeneration:
    """Factory function to create and generate report."""
    reporting_config = cfg_to_dict(config.reporting)
    target_path = str(reporting_config.get("_target_", ""))
    resolved_target = resolve_target(target_path, _REPORTING_PREFIX)

    try:
        reporter_class = instantiate({"_target_": resolved_target, "_partial_": True})
        reporter = reporter_class(config)
    except Exception as error:
        raitap_log.exception("Reporter instantiation failed for target %r", target_path)
        raise ValueError(
            f"Could not instantiate reporter {target_path!r}.\n"
            "Check that _target_ points to a valid BaseReporter implementation."
        ) from error

    report_path = reporter.generate(report.sections, report_dir=report.report_dir)
    manifest_path = report_path.parent / "report_manifest.json"
    report.manifest.write(manifest_path, report_dir=report_path.parent)
    archive_path = (
        _create_html_report_archive(report_path, manifest_path)
        if report_path.suffix.lower() == ".html"
        else None
    )
    raitap_log.info("Report generated: %s", report_path)

    return ReportGeneration(
        report_path=report_path,
        reporter=reporter,
        manifest_path=manifest_path,
        archive_path=archive_path,
    )


def _create_html_report_archive(report_path: Path, manifest_path: Path) -> Path:
    report_dir = report_path.parent
    archive_path = report_path.with_suffix(".zip")
    files = [report_path, manifest_path]
    assets_dir = report_dir / "_assets"
    if assets_dir.exists():
        files.extend(path for path in sorted(assets_dir.rglob("*")) if path.is_file())

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(report_dir))

    return archive_path
