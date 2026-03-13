from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import torch

from raitap.configs.schema import AppConfig, MetricsConfig
from raitap.metrics import evaluate, evaluate_and_log


def _config(tmp_path) -> AppConfig:
    cfg = AppConfig(experiment_name="test", fallback_output_dir=str(tmp_path))
    cfg.metrics = MetricsConfig(
        _target_="ClassificationMetrics",
        task="multiclass",
        num_classes=3,
    )
    return cfg


def test_evaluate_writes_outputs(tmp_path):
    cfg = _config(tmp_path)
    predictions = torch.tensor([0, 1, 2, 1])
    targets = torch.tensor([0, 1, 2, 0])

    out = evaluate(cfg, predictions, targets)

    assert "result" in out
    assert "run_dir" in out
    run_dir = out["run_dir"]
    assert run_dir == tmp_path

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "artifacts.json").exists()
    assert (run_dir / "metadata.json").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    assert "accuracy" in metrics
    assert metadata["target"] == "raitap.metrics.ClassificationMetrics"


def test_evaluate_bad_target_raises(tmp_path):
    cfg = _config(tmp_path)
    cfg.metrics._target_ = "DoesNotExist"

    with pytest.raises(ValueError, match="Could not instantiate metric"):
        evaluate(cfg, torch.tensor([0]), torch.tensor([0]))


def test_evaluate_respects_explicit_output_dir(tmp_path):
    cfg = _config(tmp_path)
    output_dir = tmp_path / "metrics"

    out = evaluate(
        cfg,
        torch.tensor([0, 1, 2, 1]),
        torch.tensor([0, 1, 2, 0]),
        output_dir=output_dir,
    )

    assert out["run_dir"] == output_dir
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "artifacts.json").exists()
    assert (output_dir / "metadata.json").exists()


def test_evaluate_and_log_uses_logger_for_metrics_and_artifacts(tmp_path):
    cfg = _config(tmp_path)
    output_dir = tmp_path / "metrics"
    logger = MagicMock()

    out = evaluate_and_log(
        cfg,
        torch.tensor([0, 1, 2, 1]),
        torch.tensor([0, 1, 2, 0]),
        logger=logger,
        output_dir=output_dir,
    )

    assert out["run_dir"] == output_dir
    logger.log_metrics.assert_called_once()
    logger.log_artifacts.assert_called_once_with(output_dir, artifact_path="metrics")
