from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class MetricResult:
    metrics: dict[str, float]
    artifacts: dict[str, Any] = field(default_factory=dict)


class MetricComputer(Protocol):
    """
    Contract:
    - All metric computers must implement `reset`, `update`, and `compute` methods.
    - `reset()` should reinitialize the state of the metric computer.
    - `update(predictions, targets)` should ingest new data for computation.
    - `compute()` should return a `MetricResult` object containing metrics and artifacts.
    """

    def reset(self) -> None: ...
    def update(self, predictions: Any, targets: Any) -> None: ...
    def compute(self) -> MetricResult: ...
