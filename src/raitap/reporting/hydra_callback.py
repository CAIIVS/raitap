from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import OmegaConf

from raitap import raitap_log

from .builder import build_merged_report
from .factory import create_report, reporting_enabled
from .manifest import ReportManifest


class ReportingSweepCallback(Callback):
    """Generate a merged report after a Hydra multirun completes."""

    def on_multirun_end(self, config: Any, **kwargs: Any) -> None:
        del kwargs
        if not reporting_enabled(config):
            raitap_log.debug("Reporting disabled; skipping merged sweep report.")
            return

        reporting_cfg = config.reporting
        if not bool(getattr(reporting_cfg, "multirun_report", True)):
            # The callback can still be registered for reporting-enabled configs; this flag
            # disables only the merged multirun report, not normal per-run reports.
            raitap_log.debug("reporting.multirun_report=false; skipping merged sweep report.")
            return

        sweep_dir_value = OmegaConf.select(config, "hydra.sweep.dir")
        if not sweep_dir_value:
            raitap_log.debug("Hydra sweep dir unavailable; skipping merged report generation.")
            return

        sweep_dir = Path(str(sweep_dir_value))
        if not sweep_dir.exists():
            raitap_log.debug(
                "Hydra sweep dir %s does not exist; skipping merged report.", sweep_dir
            )
            return

        child_manifests: list[tuple[str, str | None, ReportManifest]] = []
        skipped_children: list[str] = []
        for child_dir in sorted(
            (path for path in sweep_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda item: int(item.name),
        ):
            manifest_path = child_dir / "reports" / "report_manifest.json"
            if not manifest_path.exists():
                skipped_children.append(child_dir.name)
                continue
            child_manifests.append(
                (
                    f"Job {child_dir.name}",
                    _override_summary(child_dir),
                    ReportManifest.load(manifest_path),
                )
            )

        if not child_manifests:
            raitap_log.info(
                "No child report manifests found in %s; skipping merged report.", sweep_dir
            )
            return

        merged_report = build_merged_report(
            config,
            sweep_dir=sweep_dir,
            child_manifests=child_manifests,
            skipped_children=skipped_children,
        )
        create_report(config, merged_report)


def _override_summary(child_dir: Path) -> str | None:
    overrides_path = child_dir / ".hydra" / "overrides.yaml"
    if not overrides_path.exists():
        return None
    try:
        loaded = OmegaConf.load(overrides_path)
    except Exception:
        raitap_log.debug("Failed to load Hydra overrides from %s", overrides_path, exc_info=True)
        return None
    if not isinstance(loaded, list):
        return None
    filtered = [str(item) for item in loaded if str(item) and not str(item).startswith("hydra.")]
    if not filtered:
        return None
    return ", ".join(filtered[:2])
