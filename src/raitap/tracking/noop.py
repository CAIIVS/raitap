from __future__ import annotations

from typing import Any

from .base import AssessmentContext


class NoopTracker:
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
        del model, artifact_path

    def log_dataset(self, dataset_info: dict[str, Any], artifact_path: str = "dataset") -> None:
        del dataset_info, artifact_path

    def log_transparency(self, results: dict[str, Any]) -> None:
        del results

    def log_metrics(self, result: dict[str, Any], prefix: str = "performance") -> None:
        del result, prefix

    def finalize(self, status: str = "FINISHED") -> None:
        del status
