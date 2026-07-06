"""Family decorator for input-parser adapters. Mirrors label_parsers/registration.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import AdapterDecoratorOptions, FamilyConfig, _register_core
from raitap.configs.schema import InputsConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.data.input_parsers.base import InputParser

INPUTS = FamilyConfig(group="data/inputs", schema=InputsConfig, package_style="flat")

T = TypeVar("T", bound="InputParser")


def input_parser(**common: Unpack[AdapterDecoratorOptions]) -> Callable[[type[T]], type[T]]:
    """Register an input-parser adapter. ``registry_name`` required."""
    common.setdefault("extra", "")

    def wrap(cls: type[T]) -> type[T]:
        return _register_core(cls, family=INPUTS, **common)

    return wrap
