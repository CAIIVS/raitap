from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from raitap.configs.schema import AppConfig, TrackingConfig
from raitap.tracking.factory import Tracker
from raitap.tracking.mlflow import MLFlowTracker


def test_tracker_returns_none_when_tracking_disabled() -> None:
    config = cast(
        "AppConfig",
        SimpleNamespace(tracking=TrackingConfig(enabled=False)),
    )
    tracker = Tracker(config)

    assert tracker is None


def test_tracker_returns_mlflow_tracker_when_tracking_enabled() -> None:
    config = cast(
        "AppConfig",
        SimpleNamespace(
            tracking=TrackingConfig(enabled=True, tracking_uri="http://127.0.0.1:5000")
        ),
    )
    tracker = Tracker(config)

    assert isinstance(tracker, MLFlowTracker)
