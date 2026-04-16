from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from hydra.utils import instantiate

from raitap.configs import cfg_to_dict, resolve_target

_TRACKING_PREFIX = "raitap.tracking."
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from raitap.configs.schema import AppConfig
    from raitap.models.backend import ModelBackend


class BaseTracker(ABC):
    @staticmethod
    def create_tracker(config: AppConfig) -> BaseTracker:
        tracking_config = cfg_to_dict(config.tracking)
        target_path = str(tracking_config.get("_target_", ""))
        resolved_target = resolve_target(target_path, _TRACKING_PREFIX)

        try:
            tracker_class = instantiate({"_target_": resolved_target, "_partial_": True})
            tracker = tracker_class(config)
        except Exception as error:
            logger.exception("Tracker instantiation failed for target %r", target_path)
            raise ValueError(
                f"Could not instantiate tracker {target_path!r}.\n"
                "Check that _target_ points to a valid TrackerProtocol implementation."
            ) from error

        return tracker

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self.terminate(successfully=exc_type is None)
        return False  # Don't suppress exceptions

    @abstractmethod
    def log_config(self) -> None: ...

    @abstractmethod
    def log_model(self, model: ModelBackend | nn.Module) -> None: ...

    @abstractmethod
    def log_dataset(self, description: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_artifacts(
        self, source_directory: str | Path | None, target_subdirectory: str | None = None
    ) -> None: ...

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        prefix: str = "performance",
    ) -> None: ...

    @abstractmethod
    def terminate(self, successfully: bool = True) -> None: ...


@runtime_checkable
class Trackable(Protocol):
    """
    Interface for objects that can be logged to a tracker.

    Mandates a ``log`` method that accepts a tracker and optional keyword arguments.
    """

    @abstractmethod
    def log(self, tracker: BaseTracker, **kwargs: Any) -> None:
        """Log the object's artifacts or metadata to the provided tracker."""
        pass
