"""Family decorator for tracking adapters.

Adapter sites use ``@register_tracker(...)`` instead of the legacy
``class Foo(BaseTracker, registry_name=..., extra=..., ...)`` class-kwargs
syntax. ``registry_name`` is enforced as required at the call site via
``Required[str]`` in ``_CommonRegKwargs`` — pyright errors if it is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import FamilyConfig, _CommonRegKwargs, _register_core
from raitap.configs.schema import TrackingConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.tracking.base_tracker import BaseTracker

TRACKING = FamilyConfig(
    group="tracking",
    schema=TrackingConfig,
    package_style="flat",
)

T = TypeVar("T", bound="BaseTracker")


def register_tracker(
    **common: Unpack[_CommonRegKwargs],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a tracker.

    ``registry_name`` is required. Tracking has no per-family required
    metadata beyond the common kwargs.
    """

    def wrap(cls: type[T]) -> type[T]:
        return _register_core(cls, family=TRACKING, **common)

    return wrap
