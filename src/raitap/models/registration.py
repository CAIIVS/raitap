"""Metadata-only decorator for model backends.

Backends are not Hydra-config adapters (no registry, no entry-point plugins);
this decorator only sets and type-checks the required ``provides``
class constant at the decoration site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Required, TypedDict, TypeVar, Unpack

if TYPE_CHECKING:
    from collections.abc import Callable

    # NOTE: ``ModelBackend`` looks unused to static scanners (CodeQL), but the
    # string TypeVar bound below needs it — without the bound, ``cls`` is
    # ``type[object]`` and ``cls.provides = ...`` fails pyright
    # (reportAttributeAccessIssue). A runtime import would cycle (backend.py
    # imports this module for ``@register``), hence the string bound. Do not
    # remove.
    from raitap.models.backend import ModelBackend
    from raitap.types import Capability

B = TypeVar("B", bound="ModelBackend")


class _BackendRegKwargs(TypedDict):
    provides: Required[frozenset[Capability]]


def register(**kwargs: Unpack[_BackendRegKwargs]) -> Callable[[type[B]], type[B]]:
    """Register a model backend. ``provides`` is required."""

    def wrap(cls: type[B]) -> type[B]:
        cls.provides = kwargs["provides"]
        return cls

    return wrap
