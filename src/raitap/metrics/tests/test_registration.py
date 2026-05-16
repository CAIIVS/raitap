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
    # abstract=True skips AdapterMixin pre-validation/registration so the
    # decorator is the SOLE registrar — the assertion below only passes if
    # `register_metrics_adapter` actually ran.
    class _StubMetric(BaseMetricComputer, abstract=True):
        def __init__(self) -> None: ...

        def reset(self) -> None:
            return None

        def update(self, predictions: Any, targets: Any) -> None:
            del predictions, targets

        def compute(self) -> MetricResult:
            return MetricResult(metrics={})

    assert "_stub_metric" in _BUILDERS["metrics"]
    assert ADAPTER_EXTRAS["_StubMetric"] == "_stub_extra"
