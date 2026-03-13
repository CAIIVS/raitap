from __future__ import annotations

from typing import Any

from ..configs.factory_utils import cfg_to_dict
from .base import AssessmentContext, Tracker
from .mlflow import MLFlowTracker
from .noop import NoopTracker


def create_tracker(config: Any) -> Tracker:
    tracking_config = cfg_to_dict(config)

    if not tracking_config.get("enabled", False):
        return NoopTracker()

    return MLFlowTracker(
        tracking_uri=tracking_config.get("tracking_uri"),
        registry_uri=tracking_config.get("registry_uri"),
        log_model=bool(tracking_config.get("log_model", False)),
        registry_enabled=bool(tracking_config.get("registry_enabled", False)),
        registered_model_name=tracking_config.get("registered_model_name"),
    )


__all__ = [
    "AssessmentContext",
    "NoopTracker",
    "Tracker",
    "create_tracker",
]
