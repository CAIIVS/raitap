from __future__ import annotations

from .base_tracker import BaseTracker
from .stop import run_stop_command

__all__ = ["BaseTracker", "run_stop_command"]

try:
    from .mlflow_tracker import MLFlowTracker

    __all__.append("MLFlowTracker")  # noqa: PYI056
except ImportError:
    pass


def __getattr__(name: str) -> object:
    """Resolve hydra-zen builders by registry name."""
    from raitap._adapters import lookup

    return lookup("tracking", name)
