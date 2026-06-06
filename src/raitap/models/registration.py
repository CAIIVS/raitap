"""Decorator + lookups for model backends.

The ``@register`` decorator sets each backend's ``provides`` and ``extensions``
class constants at the decoration site and indexes the class by file extension,
so ``model._load_from_path`` can resolve a backend from a model file's suffix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, Required, TypedDict, TypeVar, Unpack

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Set as AbstractSet

    # NOTE: ``ModelBackend`` looks unused to static scanners (CodeQL), but the
    # string TypeVar bound below needs it — without the bound, ``cls`` is
    # ``type[object]`` and ``cls.provides = ...`` fails pyright
    # (reportAttributeAccessIssue). A runtime import would cycle (backend.py
    # imports this module for ``@register``), hence the string bound. Do not
    # remove.
    from raitap.models.backend import ModelBackend
    from raitap.types import Capability

B = TypeVar("B", bound="ModelBackend")

#: extension (e.g. ``".onnx"``) -> backend class. Populated by ``@register``.
_BACKENDS_BY_EXTENSION: dict[str, type] = {}


class _BackendRegKwargs(TypedDict):
    provides: Required[AbstractSet[Capability]]
    extensions: NotRequired[AbstractSet[str]]


def register(**kwargs: Unpack[_BackendRegKwargs]) -> Callable[[type[B]], type[B]]:
    """Register a model backend. Sets ``provides`` + ``extensions`` and indexes
    the class by extension for resolution in ``model._load_from_path``."""

    def wrap(cls: type[B]) -> type[B]:
        cls.provides = frozenset(kwargs["provides"])
        exts = frozenset(e.lower() for e in kwargs.get("extensions", frozenset()))
        cls.extensions = exts
        for ext in exts:
            _BACKENDS_BY_EXTENSION[ext] = cls
        return cls

    return wrap


def backend_for_extension(suffix: str) -> type | None:
    """Return the backend class registered for ``suffix`` (e.g. ``".onnx"``), or None."""
    return _BACKENDS_BY_EXTENSION.get(suffix.lower())


def supported_model_formats() -> list[str]:
    """Sorted list of registered file extensions."""
    return sorted(_BACKENDS_BY_EXTENSION)
