from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from raitap.configs.schema import AppConfig

    from .sections import ReportSection


class BaseReporter(ABC):
    """Abstract base class for report generators."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, sections: Sequence[ReportSection]) -> Path:
        """Generate report and return path to output file.

        Args:
            sections: Ordered report sections (metrics, transparency, etc.).

        Returns:
            Path to generated report file.
        """
        ...
