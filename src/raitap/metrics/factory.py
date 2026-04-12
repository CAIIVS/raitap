"""Config helpers for building metric computers (mirrors transparency factory style)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hydra.utils import instantiate

from raitap.configs import cfg_to_dict, resolve_run_dir, resolve_target
from raitap.utils.serialization import to_json_serialisable

if TYPE_CHECKING:
    from pathlib import Path

    from ..configs.schema import AppConfig
    from ..tracking.base_tracker import BaseTracker

from .base_metric import BaseMetricComputer, MetricResult, scalar_metrics_for_tracking
from .visualizers import MetricsVisualizer

logger = logging.getLogger(__name__)

_METRICS_PREFIX = "raitap.metrics."


def metrics_run_enabled(config: AppConfig) -> bool:
    """True when ``metrics`` is present and ``_target_`` is a non-empty string."""
    metrics_cfg = getattr(config, "metrics", None)
    if metrics_cfg is None:
        return False
    target = getattr(metrics_cfg, "_target_", None)
    if target is None:
        return False
    return bool(str(target).strip())


def create_metric(metrics_config: Any) -> tuple[BaseMetricComputer, str]:
    """Instantiate a metric computer from Hydra-style config (``_target_`` + kwargs)."""
    metrics_cfg = cfg_to_dict(metrics_config)
    target_path: str = metrics_cfg.get("_target_", "")
    resolved_target = resolve_target(target_path, _METRICS_PREFIX)
    metrics_cfg["_target_"] = resolved_target

    try:
        metric = instantiate(metrics_cfg)
    except Exception as e:
        logger.exception("Metric instantiation failed for target %r", target_path)
        raise ValueError(
            f"Could not instantiate metric {target_path!r}.\n"
            "Check that _target_ points to a valid MetricComputer implementation."
        ) from e

    return metric, resolved_target


@dataclass
class MetricsEvaluation:
    """Outcome of a metrics run (JSON on disk + optional computer handle)."""

    result: MetricResult
    run_dir: Path
    computer: BaseMetricComputer
    resolved_target: str

    def log(self, tracker: BaseTracker | None, *, prefix: str = "performance") -> None:
        if tracker is None:
            return
        scalars = scalar_metrics_for_tracking(self.result)
        if scalars:
            tracker.log_metrics(scalars, prefix=prefix)
        tracker.log_artifacts(self.run_dir, target_subdirectory="metrics")

    def create_visualizations(self) -> dict[str, Any]:
        """Generate matplotlib figures for metrics."""

        return MetricsVisualizer.create_figures(self.result)


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
            json.dumps(to_json_serialisable(result.metrics), indent=2),
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
                k: to_json_serialisable(v) for k, v in metrics_cfg.items() if k != "_target_"
            },
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info("Metrics saved: %s/metrics.json", run_dir)
        logger.info("Artifacts saved: %s/artifacts.json", run_dir)
        logger.info("Metadata saved: %s/metadata.json", run_dir)

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
