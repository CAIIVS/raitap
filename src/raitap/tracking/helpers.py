from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.tracking.base_tracker import BaseTracker


def log_model_artifact(
    tracker: BaseTracker | None,
    model: Any,
    artifact_path: str = "model",
) -> None:
    if tracker is None:
        return
    tracker.log_model(model, artifact_path=artifact_path)


def log_dataset_info(
    tracker: TrackerProtocol | None,
    config: Any,
    data: Any,
    artifact_path: str = "dataset",
) -> None:
    if tracker is None:
        return

    tracker.log_dataset(
        describe_data(
            data,
            name=config.data.name,
            source=config.data.source,
        ),
        artifact_path=artifact_path,
    )


def log_artifact_directory(
    tracker: TrackerProtocol | None,
    local_dir: str | Path,
    artifact_path: str,
) -> None:
    if tracker is None:
        return
    tracker.log_artifacts(local_dir, artifact_path=artifact_path)
