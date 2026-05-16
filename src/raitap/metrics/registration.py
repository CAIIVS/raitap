"""Family decorator for metrics adapters.

Adapter sites use ``@register_metrics_adapter(...)`` instead of the legacy
``class Foo(BaseMetricComputer, registry_name=..., extra=..., ...)`` class-kwargs
syntax. ``registry_name`` is type-checked by pyright at the decoration site via
``Required[str]`` in ``_CommonRegKwargs``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import FamilyConfig, _CommonRegKwargs, _register_core
from raitap.configs.schema import MetricsConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.metrics.base_metric_computer import BaseMetricComputer

METRICS = FamilyConfig(
    group="metrics",
    schema=MetricsConfig,
    package_style="flat",
    strip_suffixes=("Metrics", "MetricComputer"),
)

T = TypeVar("T", bound="BaseMetricComputer")


def register_metrics_adapter(
    **common: Unpack[_CommonRegKwargs],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a metrics adapter.

    ``registry_name`` is required (enforced via ``Required[str]`` in
    ``_CommonRegKwargs``). Metrics has no per-family required metadata
    beyond the common kwargs.
    """

    def wrap(cls: type[T]) -> type[T]:
        return _register_core(cls, family=METRICS, **common)

    return wrap
