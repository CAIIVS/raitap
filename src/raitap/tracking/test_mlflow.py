from __future__ import annotations

from unittest.mock import MagicMock

from raitap.tracking.base import AssessmentContext
from raitap.tracking.mlflow import MLFlowTracker


def test_start_assessment_uses_main_experiment_name(monkeypatch):
    mlflow_mock = MagicMock()
    tracker = MLFlowTracker(tracking_uri="http://127.0.0.1:5000")

    monkeypatch.setattr(tracker, "_require_mlflow", lambda: mlflow_mock)

    tracker.start_assessment(
        AssessmentContext(
            assessment_name="audit_2026_q1",
            model_source="resnet50",
            data_name="imagenet_samples",
        )
    )

    mlflow_mock.set_tracking_uri.assert_called_once_with("http://127.0.0.1:5000")
    mlflow_mock.set_experiment.assert_called_once_with("audit_2026_q1")
    mlflow_mock.start_run.assert_called_once_with(run_name="audit_2026_q1")


def test_log_artifacts_logs_existing_directory(monkeypatch, tmp_path):
    mlflow_mock = MagicMock()
    tracker = MLFlowTracker()
    artifact_dir = tmp_path / "transparency"
    artifact_dir.mkdir()

    monkeypatch.setattr(tracker, "_require_mlflow", lambda: mlflow_mock)

    tracker.log_artifacts(artifact_dir, artifact_path="transparency")

    mlflow_mock.log_artifacts.assert_called_once_with(
        str(artifact_dir),
        artifact_path="transparency",
    )


def test_log_metrics_logs_scalar_metrics_with_prefix(monkeypatch):
    mlflow_mock = MagicMock()
    tracker = MLFlowTracker()

    monkeypatch.setattr(tracker, "_require_mlflow", lambda: mlflow_mock)

    tracker.log_metrics({"accuracy": 1.0, "count": 4, "flag": True})

    mlflow_mock.log_metrics.assert_called_once_with(
        {
            "performance.accuracy": 1.0,
            "performance.count": 4.0,
            "performance.flag": 1.0,
        }
    )


def test_finalize_ends_run_with_failed_status(monkeypatch):
    mlflow_mock = MagicMock()
    tracker = MLFlowTracker()
    tracker._active_run = True

    monkeypatch.setattr(tracker, "_require_mlflow", lambda: mlflow_mock)

    tracker.finalize(status="FAILED")

    mlflow_mock.end_run.assert_called_once_with(status="FAILED")
    assert tracker._active_run is False


def test_finalize_without_active_run_is_noop(monkeypatch):
    mlflow_mock = MagicMock()
    tracker = MLFlowTracker()

    monkeypatch.setattr(tracker, "_require_mlflow", lambda: mlflow_mock)

    tracker.finalize(status="FAILED")

    mlflow_mock.end_run.assert_not_called()
