"""Decorator integration test: a decorated stub tracker must land in
``_BUILDERS`` under the ``tracking`` group with its ``extra`` recorded."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap import adapters
from raitap.tracking.base_tracker import BaseTracker

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from raitap.models.base_backend import ModelBackend


def test_tracker_registers_under_tracking_group() -> None:
    @adapters.tracker(
        registry_name="_stub_tracker",
        extra="_stub_extra",
    )
    class _StubTracker(BaseTracker):
        def __init__(self) -> None:
            pass

        def log_config(self) -> None:
            return None

        def log_model(self, model: ModelBackend | nn.Module) -> None:
            del model

        def log_dataset(self, description: dict[str, Any]) -> None:
            del description

        def log_artifacts(
            self,
            source_directory: str | Path | None,
            target_subdirectory: str | None = None,
        ) -> None:
            del source_directory, target_subdirectory

        def log_metrics(self, metrics: dict[str, float], prefix: str = "performance") -> None:
            del metrics, prefix

        def terminate(self, successfully: bool = True) -> None:
            del successfully

    from raitap._adapters import _BUILDERS, ADAPTER_EXTRAS

    assert "_stub_tracker" in _BUILDERS["tracking"]
    assert ADAPTER_EXTRAS["_StubTracker"] == "_stub_extra"
