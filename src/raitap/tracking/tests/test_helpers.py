from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

from raitap.tracking.helpers import (
    finalize_tracking,
    initialize_tracking,
    log_artifact_directory,
    log_dataset_info,
    log_model_artifact,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        experiment_name="audit_2026_q1",
        model=SimpleNamespace(source="resnet50"),
        data=SimpleNamespace(name="imagenet_samples", source="/tmp/imagenet"),
    )


def test_initialize_tracking_starts_run_and_logs_config(tmp_path: Path) -> None:
    tracker = MagicMock()
    config = _config()

    initialize_tracking(tracker, config)

    tracker.start_assessment.assert_called_once_with("audit_2026_q1")
    tracker.log_config.assert_called_once_with(config)


def test_log_dataset_info_builds_dataset_metadata() -> None:
    tracker = MagicMock()
    config = _config()
    data = SimpleNamespace(shape=(4, 3, 224, 224), dtype="float32")

    log_dataset_info(tracker, config, data)

    tracker.log_dataset.assert_called_once_with(
        {
            "name": "imagenet_samples",
            "source": "/tmp/imagenet",
            "num_samples": 4,
            "shape": [4, 3, 224, 224],
            "sample_shape": [3, 224, 224],
            "dtype": "float32",
        },
        artifact_path="dataset",
    )


def test_optional_tracking_helpers_noop_on_none(tmp_path: Path) -> None:
    data = SimpleNamespace(shape=(2, 8), dtype="float32")

    initialize_tracking(None, _config())
    log_model_artifact(None, object())
    log_dataset_info(None, _config(), data)
    log_artifact_directory(None, "artifacts", artifact_path="metrics")
    finalize_tracking(None, status="FAILED")
