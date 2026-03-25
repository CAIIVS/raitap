from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    metrics: dict[str, float]
    artifacts: dict[str, Any] = field(default_factory=dict)


def scalar_metrics_for_tracking(result: MetricResult) -> dict[str, float | int | bool]:
    """Keep only JSON-friendly scalars suitable for tracker ``log_metrics``."""
    metrics = result.metrics
    return {
        str(key): value for key, value in metrics.items() if isinstance(value, (int, float, bool))
    }


class BaseMetricComputer(ABC):
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def update(self, predictions: Any, targets: Any) -> None: ...
    @abstractmethod
    def compute(self) -> MetricResult: ...
