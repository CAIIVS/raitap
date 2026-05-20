"""Metadata-only decorator for model backends. (Body completed in Task A4.)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Required, TypedDict, TypeVar, Unpack

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.models.backend import ModelBackend

B = TypeVar("B", bound="ModelBackend")


class _BackendRegKwargs(TypedDict):
    supports_torch_autograd: Required[bool]


def register(**kwargs: Unpack[_BackendRegKwargs]) -> Callable[[type[B]], type[B]]:
    """Register a model backend. ``supports_torch_autograd`` is required."""

    def wrap(cls: type[B]) -> type[B]:
        cls.supports_torch_autograd = kwargs["supports_torch_autograd"]
        return cls

    return wrap
