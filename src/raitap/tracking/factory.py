from __future__ import annotations

from typing import Any

from ..configs.factory_utils import cfg_to_dict
from .base import AssessmentContext, Tracker
from .noop import NoopTracker


def create_tracker(config: Any) -> Tracker:
    tracking_config = cfg_to_dict(config)

    if not tracking_config.get("enabled", False):
        return NoopTracker()

    raise NotImplementedError("MLFlow tracking not yet implemented.")


__all__ = [
    "AssessmentContext",
    "NoopTracker",
    "Tracker",
    "create_tracker",
]
