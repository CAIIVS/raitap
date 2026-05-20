from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from raitap import adapters
from raitap._adapters import _BUILDERS, ADAPTER_EXTRAS
from raitap.metrics.base_metric_computer import BaseMetricComputer, MetricResult


@dataclass
class _DummyConfig:
    _target_: str = ""
    knob: int = 42


@adapters.metrics(registry_name="dummy_schema_test", schema=_DummyConfig)
class _DummyMetric(BaseMetricComputer):
    def __init__(self, *, knob: int = 0) -> None:
        self.knob = knob

    def reset(self) -> None:
        return None

    def update(self, predictions: Any, targets: Any) -> None:
        del predictions, targets

    def compute(self) -> MetricResult:
        return MetricResult(metrics={"knob": float(self.knob)})


# Module-level registration is required so hydra-zen's ``builds(...)`` can
# resolve a stable qualname for ``_DummyMetric`` (function-local classes hit
# the TypeError fallback in ``_register_core`` and never reach ``_BUILDERS``).
# That however leaks into ``ADAPTER_EXTRAS`` which the static scanner refuses
# to look at (it skips ``tests/``), so drop it here to keep
# ``test_runtime_extras_subset_of_static_scan`` honest.
ADAPTER_EXTRAS.pop("_DummyMetric", None)


def test_per_adapter_schema_used_for_builder() -> None:
    builder = _BUILDERS["metrics"]["dummy_schema_test"]
    field_names = {f.name for f in dataclasses.fields(builder)}
    assert "knob" in field_names
