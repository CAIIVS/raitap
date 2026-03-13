from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from raitap.tracking.helpers import (
    finalize_tracking,
    initialize_tracking,
    log_artifact_directory,
    log_dataset_info,
    log_model_artifact,
)


def _config():
    return SimpleNamespace(
        experiment_name="audit_2026_q1",
        model=SimpleNamespace(source="resnet50"),
        data=SimpleNamespace(name="imagenet_samples", source="/tmp/imagenet"),
    )


def test_initialize_tracking_starts_run_and_logs_config(tmp_path):
    tracker = MagicMock()
    config = _config()
    output_dir = tmp_path / "outputs"

    initialize_tracking(tracker, config, output_dir)

    tracker.start_assessment.assert_called_once()
    context = tracker.start_assessment.call_args.args[0]
    assert context.assessment_name == "audit_2026_q1"
    assert context.model_source == "resnet50"
    assert context.data_name == "imagenet_samples"
    assert context.data_source == "/tmp/imagenet"
    assert context.output_dir == output_dir
    tracker.log_config.assert_called_once_with(config)


def test_log_dataset_info_builds_dataset_metadata():
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


def test_optional_tracking_helpers_noop_on_none(tmp_path):
    data = SimpleNamespace(shape=(2, 8), dtype="float32")

    initialize_tracking(None, _config(), tmp_path)
    log_model_artifact(None, object())
    log_dataset_info(None, _config(), data)
    log_artifact_directory(None, Path("artifacts"), artifact_path="metrics")
    finalize_tracking(None, status="FAILED")
