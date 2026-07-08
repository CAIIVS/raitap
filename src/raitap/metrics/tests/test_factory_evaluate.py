from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from raitap.configs import set_output_root
from raitap.configs.registry_resolve import UnsafeConfigTargetError
from raitap.configs.schema import (
    AppConfig,
    MetricsConfig,
    MulticlassClassificationMetricsConfig,
)
from raitap.metrics import MetricsEvaluation, evaluate, metrics_run_enabled
from raitap.metrics.base_metric_computer import BaseMetricComputer, MetricResult
from raitap.metrics.factory import create_metric


def test_metrics_run_enabled_respects_empty_use(tmp_path: Path) -> None:
    cfg = AppConfig(experiment_name="t")
    set_output_root(cfg, tmp_path)
    assert not metrics_run_enabled(cfg)  # metrics is None by default
    cfg.metrics = MetricsConfig(use="")
    assert not metrics_run_enabled(cfg)
    cfg.metrics = MetricsConfig(use="multiclass_classification")
    assert metrics_run_enabled(cfg)


def test_metrics_run_enabled_rejects_config_target(tmp_path: Path) -> None:
    """A `_target_`-carrying block must raise loudly, not read as "not configured"."""
    cfg = AppConfig(experiment_name="t")
    set_output_root(cfg, tmp_path)
    cfg.metrics = {"_target_": "os.system"}  # type: ignore[assignment]

    with pytest.raises(UnsafeConfigTargetError):
        metrics_run_enabled(cfg)


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(experiment_name="test")
    set_output_root(cfg, tmp_path)
    cfg.metrics = MulticlassClassificationMetricsConfig(num_classes=3)
    return cfg


def test_evaluate_writes_outputs(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    predictions = torch.tensor([0, 1, 2, 1])
    targets = torch.tensor([0, 1, 2, 0])

    out = evaluate(cfg, predictions, targets)

    assert isinstance(out, MetricsEvaluation)
    run_dir = out.run_dir
    assert run_dir == tmp_path / "metrics"

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "artifacts.json").exists()
    assert (run_dir / "metadata.json").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    assert "accuracy" in metrics
    expected_target = "raitap.metrics.classification_metrics.MulticlassClassificationMetrics"
    assert metadata["target"] == expected_target
    assert "use" not in metadata["metric_config"]
    assert "_target_" not in metadata["metric_config"]


def test_evaluate_bad_use_raises(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    assert cfg.metrics is not None
    cfg.metrics.use = "does_not_exist"

    with pytest.raises(ValueError, match="Unknown metrics key"):
        evaluate(cfg, torch.tensor([0]), torch.tensor([0]))


def test_create_metric_rejects_config_target() -> None:
    with pytest.raises(UnsafeConfigTargetError):
        create_metric({"_target_": "os.system", "use": "multiclass_classification"})


def test_create_metric_rejects_unknown_use() -> None:
    with pytest.raises(ValueError, match="Unknown metrics key"):
        create_metric({"use": "does_not_exist"})


def test_evaluate_writes_under_metrics_subdirectory(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    out = evaluate(cfg, torch.tensor([0, 1, 2, 1]), torch.tensor([0, 1, 2, 0]))

    assert out.run_dir == tmp_path / "metrics"
    assert (tmp_path / "metrics" / "metrics.json").exists()
    assert (tmp_path / "metrics" / "artifacts.json").exists()
    assert (tmp_path / "metrics" / "metadata.json").exists()


def test_metrics_evaluation_log_uses_logger_for_metrics_and_artifacts(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    logger = MagicMock()

    out = evaluate(cfg, torch.tensor([0, 1, 2, 1]), torch.tensor([0, 1, 2, 0]))
    out.log(logger)

    assert out.run_dir == tmp_path / "metrics"
    logger.log_metrics.assert_called_once()
    logger.log_artifacts.assert_called_once_with(
        tmp_path / "metrics", target_subdirectory="metrics"
    )


def test_evaluate_prepares_metric_inputs_before_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeMetric(BaseMetricComputer):
        def __init__(self) -> None:
            self.prepare_called_with: tuple[object, object] | None = None
            self.update_called_with: tuple[object, object] | None = None

        def _prepare_inputs(self, predictions: object, targets: object) -> tuple[object, object]:
            self.prepare_called_with = (predictions, targets)
            return ("prepared_predictions", "prepared_targets")

        def _move_to_device(self, device: torch.device | None) -> None:
            del device
            return None

        def reset(self) -> None:
            return None

        def update(self, predictions: object, targets: object) -> None:
            self.update_called_with = (predictions, targets)

        def compute(self) -> MetricResult:
            return MetricResult(scalars={"accuracy": 1.0})

    cfg = _config(tmp_path)
    fake_metric = FakeMetric()
    predictions = torch.tensor([0, 1, 2, 1])
    targets = torch.tensor([0, 1, 2, 0])

    monkeypatch.setattr(
        "raitap.metrics.factory.create_metric",
        lambda _metrics_cfg: (fake_metric, "raitap.metrics.FakeMetric"),
    )

    evaluate(cfg, predictions, targets)

    assert fake_metric.prepare_called_with == (predictions, targets)
    assert fake_metric.update_called_with == ("prepared_predictions", "prepared_targets")


def test_evaluate_metrics_dispatches_detection_metrics_on_forward_output(
    tmp_path: Path,
) -> None:
    """ForwardOutput(detection) + list[dict] labels → DetectionMetrics computes mAP."""
    from raitap.configs.schema import DetectionMetricsConfig
    from raitap.metrics.phase import evaluate_metrics
    from raitap.pipeline.outputs import ForwardOutput
    from raitap.types import TaskKind

    cfg = AppConfig(experiment_name="t")
    set_output_root(cfg, tmp_path)
    cfg.metrics = DetectionMetricsConfig()

    detection_predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
    ]
    labels = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
    ]
    forward = ForwardOutput(
        task_kind=TaskKind.detection,
        batch_size=1,
        payload=detection_predictions,
    )

    result = evaluate_metrics(cfg, forward, labels)
    assert result is not None
    assert result.resolved_target == "raitap.metrics.detection_metrics.DetectionMetrics"


def test_evaluate_metrics_skips_detection_without_labels(tmp_path: Path) -> None:
    from raitap.configs.schema import DetectionMetricsConfig
    from raitap.metrics.phase import evaluate_metrics
    from raitap.pipeline.outputs import ForwardOutput
    from raitap.types import TaskKind

    cfg = AppConfig(experiment_name="t")
    set_output_root(cfg, tmp_path)
    cfg.metrics = DetectionMetricsConfig()

    forward = ForwardOutput(
        task_kind=TaskKind.detection,
        batch_size=1,
        payload=[
            {
                "boxes": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.int64),
            }
        ],
    )
    assert evaluate_metrics(cfg, forward, None) is None
