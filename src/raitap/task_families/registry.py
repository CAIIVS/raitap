"""Core-only registry of task families.

A family registers a singleton instance under its ``kind``. The pipeline
resolves the family once from ``backend.task_kind`` and threads the object
into each phase. Not plugin-shippable (see spec D1); a clean base to add
entry-point discovery later if a real external user appears.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from raitap.task_families.base import TaskFamily
    from raitap.types import TaskKind

#: kind -> the family singleton serving it.
TASK_FAMILIES: dict[TaskKind, TaskFamily] = {}

T = TypeVar("T")


def task_family(cls: type[T]) -> type[T]:
    """Register ``cls`` (instantiated once) under its ``kind`` class attribute."""
    instance = cls()  # type: ignore[call-arg]
    TASK_FAMILIES[instance.kind] = instance  # type: ignore[attr-defined]
    return cls


def resolve_task_family(kind: TaskKind) -> TaskFamily:
    """Return the family serving ``kind``.

    A backend instance wraps exactly one model, so it carries exactly one
    task family. Raises ``KeyError`` if no family is registered for ``kind``.
    """
    return TASK_FAMILIES[kind]
