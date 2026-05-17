"""Decorator integration test: a decorated stub metric must land in _BUILDERS
under the ``metrics`` group."""

from __future__ import annotations

from typing import Any

from raitap._adapters import _BUILDERS, ADAPTER_EXTRAS
from raitap.metrics.base_metric_computer import BaseMetricComputer, MetricResult
from raitap.metrics.registration import register_metrics_adapter


def test_register_metrics_adapter_registers_under_metrics_group() -> None:
    @register_metrics_adapter(
        registry_name="_stub_metric",
        extra="_stub_extra",
    )
    class _StubMetric(BaseMetricComputer):
        def __init__(self) -> None:
            pass

        def reset(self) -> None:
            return None

        def update(self, predictions: Any, targets: Any) -> None:
            del predictions, targets

        def compute(self) -> MetricResult:
            return MetricResult(metrics={})

    assert "_stub_metric" in _BUILDERS["metrics"]
    assert ADAPTER_EXTRAS["_StubMetric"] == "_stub_extra"
