"""Config helpers for building metric computers (mirrors transparency factory style)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
from hydra.utils import instantiate

from raitap import raitap_log
from raitap.configs import cfg_to_dict, resolve_run_dir
from raitap.configs.registry_resolve import stamp_target_from_use, use_key_enabled
from raitap.reporting.sections import Reportable, ReportGroup, ReportSection
from raitap.reporting.staging import _copy_asset
from raitap.tracking.base_tracker import BaseTracker, Trackable
from raitap.utils.serialization import to_json_serialisable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from raitap.configs.schema import AppConfig
    from raitap.reporting.sections import ReportContext


from .base_metric_computer import BaseMetricComputer, MetricResult, scalar_metrics_for_tracking
from .visualizers import MetricsVisualizer


def metrics_run_enabled(config: AppConfig) -> bool:
    """True when ``metrics`` is present and ``use`` is a non-empty string."""
    metrics_cfg = config.metrics
    if metrics_cfg is None:
        return False
    return use_key_enabled(cfg_to_dict(metrics_cfg))


def create_metric(metrics_config: Any) -> tuple[BaseMetricComputer, str]:
    """Instantiate a metric computer from config (``use: <registry key>`` + kwargs)."""
    metrics_cfg = cfg_to_dict(metrics_config)
    use = str(metrics_cfg.get("use", ""))
    stamp_target_from_use(metrics_cfg, group="metrics")
    resolved_target = metrics_cfg["_target_"]

    try:
        metric = instantiate(metrics_cfg)
    except Exception as e:
        raitap_log.exception("Metric instantiation failed for target %r", resolved_target)
        raise ValueError(
            f"Could not instantiate metric {use!r}.\n"
            "Check that `use` points to a registered metrics adapter."
        ) from e

    return metric, resolved_target


@dataclass
class MetricsEvaluation(Trackable, Reportable):
    """Outcome of a metrics run (JSON on disk + optional computer handle)."""

    result: MetricResult
    run_dir: Path
    computer: BaseMetricComputer
    resolved_target: str

    def log(
        self,
        tracker: BaseTracker | None,
        *,
        prefix: str = "performance",
        **kwargs: Any,
    ) -> None:
        if tracker is None:
            return
        scalars = scalar_metrics_for_tracking(self.result)
        if scalars:
            tracker.log_metrics(scalars, prefix=prefix)
        tracker.log_artifacts(self.run_dir, target_subdirectory="metrics")

    report_order: ClassVar[int] = 10

    def to_report_group(self) -> ReportGroup:
        table_rows = tuple(
            (str(name), f"{float(value):.4f}") for name, value in self.result.scalars.items()
        )
        images = tuple(sorted(self.run_dir.glob("*.png")))
        return ReportGroup(
            heading="Performance Metrics",
            images=images,
            table_rows=table_rows,
        )

    def report_sections(self, ctx: ReportContext) -> Sequence[ReportSection]:
        source_group = self.to_report_group()
        staged_images = tuple(
            _copy_asset(
                image_path,
                assets_dir=ctx.assets_dir,
                target_name=f"metrics_{index}{image_path.suffix}",
            )
            for index, image_path in enumerate(source_group.images)
        )
        group = ReportGroup(
            heading=source_group.heading,
            images=staged_images,
            table_rows=source_group.table_rows,
            metadata={"role": "metrics"},
        )
        return (
            ReportSection.from_groups("Metrics", [group], metadata={"section_role": "metrics"}),
        )


class Metrics:
    """Run configured metrics, write JSON under the run ``metrics/`` directory."""

    def __new__(
        cls,
        config: AppConfig,
        predictions: Any,
        targets: Any,
    ) -> MetricsEvaluation:
        metric, resolved_target = create_metric(config.metrics)
        predictions, targets = metric._prepare_inputs(predictions, targets)
        metric.update(predictions, targets)
        result = metric.compute()

        run_dir = resolve_run_dir(config, subdir="metrics")
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "metrics.json").write_text(
            json.dumps(to_json_serialisable(result.scalars), indent=2),
            encoding="utf-8",
        )
        (run_dir / "artifacts.json").write_text(
            json.dumps(to_json_serialisable(result.artifacts), indent=2),
            encoding="utf-8",
        )
        metrics_cfg = cfg_to_dict(config.metrics)
        metadata = {
            "experiment_name": config.experiment_name,
            "target": resolved_target,
            "metric_config": {
                k: to_json_serialisable(v)
                for k, v in metrics_cfg.items()
                if k not in ("_target_", "use")
            },
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        raitap_log.info("Metrics saved: %s/metrics.json", run_dir)
        raitap_log.info("Artifacts saved: %s/artifacts.json", run_dir)
        raitap_log.info("Metadata saved: %s/metadata.json", run_dir)

        try:
            figures = MetricsVisualizer.create_figures(result)
            for name, fig in figures.items():
                fig.savefig(run_dir / f"{name}.png", bbox_inches="tight", dpi=150)
                plt.close(fig)
        except Exception:
            raitap_log.exception("Failed to save metric charts")

        return MetricsEvaluation(
            result=result,
            run_dir=run_dir,
            computer=metric,
            resolved_target=resolved_target,
        )


def evaluate(
    config: AppConfig,
    predictions: Any,
    targets: Any,
) -> MetricsEvaluation:
    """Compute metrics, persist JSON outputs, and return a :class:`MetricsEvaluation`."""
    return Metrics(config, predictions, targets)
