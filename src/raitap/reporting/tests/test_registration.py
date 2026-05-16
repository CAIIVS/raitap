"""Decorator integration test: a decorated stub reporter must land in _BUILDERS
under the reporting group."""

from __future__ import annotations

from raitap.reporting.base_reporter import BaseReporter
from raitap.reporting.registration import register_reporter


def test_register_reporter_registers_under_reporting_group() -> None:
    @register_reporter(
        registry_name="_stub_reporter",
        extra="_stub_extra",
        library="_stub_lib",
    )
    class _StubReporter(BaseReporter):
        def __init__(self, config=None):
            self.config = config

        def generate(self, sections, *, report_dir=None):
            del sections, report_dir
            return None  # type: ignore[return-value]

    from raitap._adapters import ADAPTER_EXTRAS, _BUILDERS

    assert "_stub_reporter" in _BUILDERS["reporting"]
    assert ADAPTER_EXTRAS["_StubReporter"] == "_stub_extra"
