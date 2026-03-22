from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.data import describe_data

if TYPE_CHECKING:
    from pathlib import Path

    from .base import Tracker


def initialize_tracking(tracker: Tracker | None, config: Any) -> None:
    if tracker is None:
        return

    tracker.start_assessment(str(config.experiment_name))
    tracker.log_config(config)


def finalize_tracking(tracker: Tracker | None, status: str = "FINISHED") -> None:
    if tracker is None:
        return
    tracker.finalize(status=status)


def log_model_artifact(
    tracker: Tracker | None,
    model: Any,
    artifact_path: str = "model",
) -> None:
    if tracker is None:
        return
    tracker.log_model(model, artifact_path=artifact_path)


def log_dataset_info(
    tracker: Tracker | None,
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
    tracker: Tracker | None,
    local_dir: str | Path,
    artifact_path: str,
) -> None:
    if tracker is None:
        return
    tracker.log_artifacts(local_dir, artifact_path=artifact_path)
