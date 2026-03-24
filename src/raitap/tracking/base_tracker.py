from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from hydra.utils import instantiate

from raitap.configs.factory_utils import cfg_to_dict, resolve_target

_TRACKING_PREFIX = "raitap.tracking."

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from raitap.configs.schema import AppConfig


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
    def log_model(self, model: nn.Module) -> None: ...

    @abstractmethod
    def log_dataset(self, description: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_artifacts(
        self, source_directory: str | Path | None, target_subdirectory: str | None = None
    ) -> None: ...

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float | int | bool],
        prefix: str = "performance",
    ) -> None: ...

    @abstractmethod
    def terminate(self, successfully: bool = True) -> None: ...
