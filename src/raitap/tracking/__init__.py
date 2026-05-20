from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base_tracker import BaseTracker
from .stop import run_stop_command

if TYPE_CHECKING:
    from raitap.configs.schema import TrackingConfig


__all__ = ["BaseTracker", "TrackingConfig", "run_stop_command"]

try:
    from .mlflow_tracker import MLFlowTracker

    __all__.append("MLFlowTracker")  # noqa: PYI056
except ImportError:
    pass


def __getattr__(name: str) -> Any:
    """Resolve hydra-zen builders by registry name, plus the schema dataclass
    (:class:`~raitap.configs.schema.TrackingConfig`) re-exported here."""
    if name == "TrackingConfig":
        from raitap.configs.schema import TrackingConfig

        return TrackingConfig
    from raitap._adapters import lookup

    try:
        return lookup("tracking", name)
    except AttributeError:
        from raitap.configs import register_configs

        register_configs()  # idempotent; fires in-tree imports + plugin discovery
        return lookup("tracking", name)
