"""Decorator integration test: a decorated stub reporter must land in _BUILDERS
under the reporting group."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from raitap.reporting.base_reporter import BaseReporter
from raitap.reporting.registration import register_reporter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from raitap.reporting.base_reporter import ReportSection


def test_register_reporter_registers_under_reporting_group() -> None:
    @register_reporter(
        registry_name="_stub_reporter",
        extra="_stub_extra",
        library="_stub_lib",
    )
    class _StubReporter(BaseReporter):
        def __init__(self, config: object = None) -> None:
            super().__init__(config)  # type: ignore[arg-type]  # stub: real BaseReporter expects AppConfig

        def generate(
            self,
            sections: Sequence[ReportSection],
            *,
            report_dir: Path | None = None,
        ) -> Path:
            del sections, report_dir
            return Path()

    from raitap._adapters import _BUILDERS, ADAPTER_EXTRAS

    assert "_stub_reporter" in _BUILDERS["reporting"]
    assert ADAPTER_EXTRAS["_StubReporter"] == "_stub_extra"
