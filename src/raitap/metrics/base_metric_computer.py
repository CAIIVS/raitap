from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from raitap._adapters import AdapterMixin
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch
else:
    torch = lazy_import("torch")


@dataclass
class MetricResult:
    scalars: dict[str, float]
    artifacts: dict[str, Any] = field(default_factory=dict)


def scalar_metrics_for_tracking(result: MetricResult) -> dict[str, float | int | bool]:
    """Keep only JSON-friendly scalars suitable for tracker ``log_metrics``."""
    return {
        str(key): value
        for key, value in result.scalars.items()
        if isinstance(value, (int, float, bool))
    }


class BaseMetricComputer(ABC, AdapterMixin):
    def _prepare_inputs(self, predictions: Any, targets: Any) -> tuple[Any, Any]:
        device = _first_tensor_device(predictions)
        if device is None:
            device = _first_tensor_device(targets)

        self._move_to_device(device)
        if device is None:
            return predictions, targets
        return _move_tensors_to_device(predictions, device), _move_tensors_to_device(
            targets, device
        )

    def _move_to_device(self, device: torch.device | None) -> None:
        del device
        return None

    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def update(self, predictions: Any, targets: Any) -> None: ...
    @abstractmethod
    def compute(self) -> MetricResult: ...


def _first_tensor_device(value: Any) -> torch.device | None:
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, dict):
        for nested in value.values():
            device = _first_tensor_device(nested)
            if device is not None:
                return device
        return None
    if isinstance(value, list | tuple):
        for nested in value:
            device = _first_tensor_device(nested)
            if device is not None:
                return device
    return None


def _move_tensors_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors_to_device(nested, device) for key, nested in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_device(nested, device) for nested in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(nested, device) for nested in value)
    return value
