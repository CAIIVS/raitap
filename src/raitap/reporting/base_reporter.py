from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig
    from raitap.metrics.factory import MetricsEvaluation
    from raitap.transparency.results import ExplanationResult


class BaseReporter(ABC):
    """Abstract base class for report generators."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(
        self,
        transparency_outputs: dict[str, ExplanationResult],
        metrics_evaluation: MetricsEvaluation | None,
    ) -> Path:
        """Generate report and return path to output file.

        Args:
            transparency_outputs: Dict mapping explainer names to ExplanationResult
            metrics_evaluation: Optional metrics evaluation results

        Returns:
            Path to generated report file
        """
        ...
