from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from hydra.utils import instantiate

from raitap import raitap_log
from raitap._adapters import AdapterMixin
from raitap.configs import cfg_to_dict, resolve_target

_TRACKING_PREFIX = "raitap.tracking."

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from raitap.configs.schema import AppConfig
    from raitap.models.backend import ModelBackend


class BaseTracker(ABC, AdapterMixin, abstract=True):
    @classmethod
    def tracker_name(cls) -> str:
        """Identity used by the process registry. Override if multiple classes
        share a single detached helper."""
        return cls.__name__

    @classmethod
    def stop_detached(cls, timeout: float = 5.0) -> tuple[int, int]:
        """Terminate this tracker's detached helpers. Returns ``(killed, skipped)``.

        Subclasses with detached helpers must override. The default
        implementation only logs an error so that misconfigured trackers
        (registry entries written without a matching ``stop_detached``
        implementation) are surfaced instead of silently leaking processes.
        """
        del timeout  # default impl has nothing to terminate
        from .process_registry import pop_entries_for_tracker, reinsert_entries

        entries, ports = pop_entries_for_tracker(cls.tracker_name())
        if not entries and not ports:
            return (0, 0)

        raitap_log.error(
            "No stop process implemented for tracker %r; %d process entries and "
            "%d watched ports left in registry.",
            cls.tracker_name(),
            len(entries),
            len(ports),
        )
        reinsert_entries(entries, ports)
        return (0, 0)

    @staticmethod
    def create_tracker(config: AppConfig) -> BaseTracker:
        tracking_config = cfg_to_dict(config.tracking)
        target_path = str(tracking_config.get("_target_", ""))
        resolved_target = resolve_target(target_path, _TRACKING_PREFIX)

        try:
            tracker_class = instantiate({"_target_": resolved_target, "_partial_": True})
            tracker = tracker_class(config)
        except Exception as error:
            raitap_log.exception("Tracker instantiation failed for target %r", target_path)
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
