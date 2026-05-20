"""Decorator integration test: a decorated stub metric must land in _BUILDERS
under the ``metrics`` group."""

from __future__ import annotations

from typing import Any

from raitap import adapters
from raitap._adapters import _BUILDERS, ADAPTER_EXTRAS
from raitap.metrics.base_metric_computer import BaseMetricComputer, MetricResult


@adapters.metrics(
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


# Module-scope registration is required so hydra-zen's ``builds(...)`` can
# resolve a stable qualname (function-local classes hit the TypeError fallback
# in ``_register_core``). Drop the resulting ``ADAPTER_EXTRAS`` entry so the
# static-scan guard stays honest — see ``test_runtime_extras_subset_of_static_scan``.
ADAPTER_EXTRAS.pop("_StubMetric", None)


def test_metrics_adapter_registers_under_metrics_group() -> None:
    assert "_stub_metric" in _BUILDERS["metrics"]
