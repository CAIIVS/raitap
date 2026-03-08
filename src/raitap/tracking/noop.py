from __future__ import annotations

from typing import Any

from .base import AssessmentContext, Tracker


class NoopTracker(Tracker):
    """
    No-op tracking implementation that does not log anything.
    This simplifies the tracking process by providing a placeholder
    for tracking operations
    """

    def start_assessment(self, context: AssessmentContext) -> None:
        del context

    def log_config(self, config: Any) -> None:
        del config

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        del model

    def log_dataset(self, dataset_info: dict[str, Any], artifact_path: str = "dataset") -> None:
        del dataset_info

    def log_transparency(self, results: dict[str, Any]) -> None:
        del results

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        del metrics

    def finalize(self, status="FINISHED") -> None:
        del status
