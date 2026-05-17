"""Family decorator for transparency visualisers (no Hydra group).

Adapter sites use ``@register_transparency_visualiser(...)`` instead of the
legacy ``class Foo(BaseVisualiser, registry_name=..., ...)`` class-kwargs
syntax. Visualisers do not own a Hydra family group — the resulting builder is
stored in ``_BUILDERS['_unscoped']`` and embedded as a list inside
``TransparencyConfig`` entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import _CommonRegKwargs, _register_core

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.transparency.visualisers.base_visualiser import BaseVisualiser

T = TypeVar("T", bound="BaseVisualiser")


def register_transparency_visualiser(
    **common: Unpack[_CommonRegKwargs],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a transparency visualiser.

    ``registry_name`` is required (enforced via ``Required[str]`` in
    ``_CommonRegKwargs``). There is no Hydra group; the builder lands in
    ``_BUILDERS['_unscoped']``.
    """

    def wrap(cls: type[T]) -> type[T]:
        return _register_core(cls, family=None, **common)

    return wrap
