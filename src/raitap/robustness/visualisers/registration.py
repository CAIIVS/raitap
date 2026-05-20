"""Family decorator for robustness visualisers (no Hydra group)."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import AdapterDecoratorOptions, _register_core

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser

T = TypeVar("T", bound="BaseRobustnessVisualiser")


def register_robustness_visualiser(
    **common: Unpack[AdapterDecoratorOptions],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a robustness visualiser. No family group; the
    builder lands in ``_BUILDERS['_unscoped']`` and is embedded as a list
    inside ``RobustnessConfig`` entries."""

    def wrap(cls: type[T]) -> type[T]:
        return _register_core(cls, family=None, **common)

    return wrap
