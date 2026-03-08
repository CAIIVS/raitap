from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class AssessmentContext:
    assessment_name: str
    model_source: str | None
    data_name: str
    data_source: str | None = None
    output_dir: Path | None = None


class Tracker(Protocol):
    def start_assessment(self, context: AssessmentContext) -> None: ...
    def log_config(self, config: Any) -> None: ...
    def log_model(self, model: Any, artifact_path: str = "model") -> None: ...
    def log_dataset(self, dataset_info: dict[str, Any], artifact_path: str = "dataset") -> None: ...
    def log_transparency(self, results: dict[str, Any]) -> None: ...
    def log_metrics(self, result: dict[str, Any], prefix: str = "performance") -> None: ...
    def finalize(self, status: str = "FINISHED") -> None: ...
