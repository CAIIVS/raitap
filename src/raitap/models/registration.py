"""Metadata-only decorator for model backends.

Backends are not Hydra-config adapters (no registry, no entry-point plugins);
this decorator only sets and type-checks the required ``supports_torch_autograd``
class constant at the decoration site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Required, TypedDict, TypeVar, Unpack

if TYPE_CHECKING:
    from collections.abc import Callable

# Unbounded: a runtime ``bound=ModelBackend`` would cycle (backend.py imports
# this module for ``@register``), and a string bound keeps a TYPE_CHECKING
# import that static scanners flag as unused. The decorator returns ``cls``
# unchanged, so the bound only documents intent — not worth the churn.
B = TypeVar("B")


class _BackendRegKwargs(TypedDict):
    supports_torch_autograd: Required[bool]


def register(**kwargs: Unpack[_BackendRegKwargs]) -> Callable[[type[B]], type[B]]:
    """Register a model backend. ``supports_torch_autograd`` is required."""

    def wrap(cls: type[B]) -> type[B]:
        cls.supports_torch_autograd = kwargs["supports_torch_autograd"]
        return cls

    return wrap
