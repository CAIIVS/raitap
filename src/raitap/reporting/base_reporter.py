from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from raitap.configs.schema import AppConfig
    from raitap.metrics.factory import MetricsEvaluation

    from .sections import ReportImageSection


class BaseReporter(ABC):
    """Abstract base class for report generators."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(
        self,
        image_sections: Sequence[ReportImageSection],
        metrics_evaluation: MetricsEvaluation | None,
    ) -> Path:
        """Generate report and return path to output file.

        Args:
            image_sections: Ordered sections of figure groups (PNG paths under ``run_dir``).
            metrics_evaluation: Optional metrics evaluation results.

        Returns:
            Path to generated report file
        """
        ...
