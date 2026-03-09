"""Factory / orchestration layer for RAITAP metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from ..configs.factory_utils import cfg_to_dict, resolve_target

if TYPE_CHECKING:
    from ..configs.schema import AppConfig

_METRICS_PREFIX = "raitap.metrics."


def _json_serialisable(value: Any) -> Any:
    """Best-effort conversion to JSON-serialisable structures."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): _json_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_serialisable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_serialisable(value.item())
        except Exception:
            pass
    return repr(value)


def evaluate(
    config: AppConfig,
    predictions: Any,
    targets: Any,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Compute metrics in one call and persist outputs to the run directory.

    Parameters
    ----------
    config:
        App config exposing ``metrics``, ``experiment_name``, and
        ``fallback_output_dir``.
    predictions:
        Model predictions
    targets:
        Ground truth
    output_dir:
        Explicit artifact directory. If omitted, Hydra's runtime output directory
        is used, falling back to ``config.fallback_output_dir``.
    """
    metrics_cfg = cfg_to_dict(config.metrics)
    target_path: str = metrics_cfg.get("_target_", "")
    metrics_cfg["_target_"] = resolve_target(target_path, _METRICS_PREFIX)

    try:
        metric = instantiate(metrics_cfg)
    except Exception as e:
        raise ValueError(
            f"Could not instantiate metric {target_path!r}.\n"
            "Check that _target_ points to a valid MetricComputer implementation."
        ) from e

    metric.update(predictions, targets)
    result = metric.compute()

    if output_dir is not None:
        run_dir = Path(output_dir)
    else:
        try:
            run_dir = Path(HydraConfig.get().runtime.output_dir)
        except ValueError:
            run_dir = Path(config.fallback_output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics.json").write_text(
        json.dumps(_json_serialisable(result.metrics), indent=2),
        encoding="utf-8",
    )
    (run_dir / "artifacts.json").write_text(
        json.dumps(_json_serialisable(result.artifacts), indent=2),
        encoding="utf-8",
    )
    metadata = {
        "experiment_name": config.experiment_name,
        "target": resolve_target(target_path, _METRICS_PREFIX),
        "metric_config": {
            k: _json_serialisable(v) for k, v in metrics_cfg.items() if k != "_target_"
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"✓ Metrics saved:   {run_dir}/metrics.json")
    print(f"✓ Artifacts saved: {run_dir}/artifacts.json")
    print(f"✓ Metadata saved:  {run_dir}/metadata.json")

    return {
        "result": result,
        "run_dir": run_dir,
    }
