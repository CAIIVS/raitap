from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hydra.utils import instantiate

from raitap.configs import cfg_to_dict, resolve_target
from raitap.tracking.base_tracker import BaseTracker, Trackable

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

    from .base_reporter import BaseReporter
    from .builder import BuiltReport


logger = logging.getLogger(__name__)
_REPORTING_PREFIX = "raitap.reporting."


def reporting_enabled(config: AppConfig) -> bool:
    """Check if reporting is enabled in config."""
    reporting_cfg = getattr(config, "reporting", None)
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
        logger.exception("Reporter instantiation failed for target %r", target_path)
        raise ValueError(
            f"Could not instantiate reporter {target_path!r}.\n"
            "Check that _target_ points to a valid BaseReporter implementation."
        ) from error

    report_path = reporter.generate(report.sections, report_dir=report.report_dir)
    manifest_path = report_path.parent / "report_manifest.json"
    report.manifest.write(manifest_path, report_dir=report_path.parent)
    logger.info("Report generated: %s", report_path)

    return ReportGeneration(
        report_path=report_path,
        reporter=reporter,
        manifest_path=manifest_path,
    )
