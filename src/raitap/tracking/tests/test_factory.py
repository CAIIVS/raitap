from __future__ import annotations

from raitap.tracking.factory import create_tracker
from raitap.tracking.mlflow import MLFlowTracker


def test_create_tracker_returns_none_when_tracking_disabled():
    tracker = create_tracker({"enabled": False})

    assert tracker is None


def test_create_tracker_returns_mlflow_tracker_when_tracking_enabled():
    tracker = create_tracker({"enabled": True, "tracking_uri": "http://127.0.0.1:5000"})

    assert isinstance(tracker, MLFlowTracker)
