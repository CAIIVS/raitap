"""Family decorator for reporting adapters.

Adapter sites use ``@register_reporter(...)`` instead of the legacy
``class Foo(BaseReporter, registry_name=..., extra=..., ...)`` class-kwargs
syntax. ``registry_name`` is required (enforced via ``Required[str]`` in
``_CommonRegKwargs``) so pyright errors at the decoration site if omitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import FamilyConfig, _CommonRegKwargs, _register_core
from raitap.configs.schema import ReportingConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.reporting.base_reporter import BaseReporter

REPORTING = FamilyConfig(
    group="reporting",
    schema=ReportingConfig,
    package_style="flat",
    strip_suffixes=("Reporter",),
)

T = TypeVar("T", bound="BaseReporter")


def register_reporter(
    **common: Unpack[_CommonRegKwargs],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a reporter.

    ``registry_name`` is required. Reporting has no per-family required
    metadata beyond the common kwargs.
    """

    def wrap(cls: type[T]) -> type[T]:
        return _register_core(cls, family=REPORTING, **common)

    return wrap
