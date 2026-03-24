from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from raitap.tracking.base_tracker import BaseTracker


@dataclass
class MetricResult:
    metrics: dict[str, float]
    artifacts: dict[str, Any] = field(default_factory=dict)


class MetricComputer(Protocol):
    def reset(self) -> None: ...
    def update(self, predictions: Any, targets: Any) -> None: ...
    def compute(self) -> MetricResult: ...
    def log(self, tracker: BaseTracker) -> None: ...
