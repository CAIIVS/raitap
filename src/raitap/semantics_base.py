"""Cross-module base interface for adapters that publish an algorithm registry.

Both ``raitap.transparency`` and ``raitap.robustness`` describe their adapters
by mapping algorithm-name → typed semantics payload (``frozenset[MethodFamily]``
for explainers, ``AssessorSemanticsHints`` for assessors). This module provides
the single shared contract:

* ``SemanticallyDescribable[T]`` — generic ABC; subclasses declare
  ``algorithm_registry: ClassVar[Mapping[str, T]]`` as a non-empty mapping.
* ``__init_subclass__`` enforces the registry at class-definition time so
  configuration mistakes fail at import, not at runtime mid-pipeline.
* ``TaskKind`` — task-family taxonomy (classification / detection / segmentation
  / seq2seq / regression) shared across the transparency and robustness modules.
* ``supported_tasks`` ClassVar on every adapter, defaulting to
  ``{TaskKind.CLASSIFICATION}`` so legacy adapters keep current behaviour
  without edits.

Intermediate abstract base classes (e.g. ``BaseAssessor``,
``EmpiricalAttackAssessor``, ``AbstractExplainer``) opt out of validation by
passing ``register=False`` in their class signature; concrete adapters always
opt in (default).
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, ClassVar, Generic, TypeVar

T = TypeVar("T")


class TaskKind(StrEnum):
    """Model task family.

    Adapters declare which task families they support via
    ``supported_tasks: ClassVar[frozenset[TaskKind]]``. The default is
    ``{CLASSIFICATION}`` so existing adapters stay correct without explicit
    declaration.
    """

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    SEQ2SEQ = "seq2seq"
    REGRESSION = "regression"


class SemanticallyDescribable(ABC, Generic[T]):
    """Adapter that publishes an algorithm-name → hints registry as a ClassVar.

    Subclasses must declare a non-empty ``algorithm_registry`` ClassVar at
    class-definition time. Pass ``register=False`` on the class line for
    abstract intermediate classes that don't (yet) carry concrete algorithms.

    The ``Generic[T]`` parameter is enforced statically only — pyright requires
    a ``# type: ignore[misc]`` on the base ClassVar declaration because PEP
    526 forbids ``ClassVar`` parameterised by a TypeVar. Concrete subclasses
    annotate ``algorithm_registry`` with the resolved type and pyright checks
    them normally.
    """

    algorithm_registry: ClassVar[Mapping[str, Any]]  # type: ignore[misc]
    """Concrete subclasses narrow the value type to ``Mapping[str, T]``."""

    supported_tasks: ClassVar[frozenset[TaskKind]] = frozenset({TaskKind.CLASSIFICATION})
    """Task families this adapter supports. Defaults to ``{CLASSIFICATION}``."""

    def __init_subclass__(cls, *, register: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        raw_tasks = cls.__dict__.get("supported_tasks")
        if raw_tasks is not None:
            if not isinstance(raw_tasks, frozenset) or not all(
                isinstance(item, TaskKind) for item in raw_tasks
            ):
                raise TypeError(
                    f"{cls.__name__}.supported_tasks must be a "
                    "frozenset[TaskKind]."
                )
            if not raw_tasks:
                raise TypeError(
                    f"{cls.__name__}.supported_tasks must contain at least "
                    "one TaskKind member."
                )
        if not register:
            return
        registry = cls.__dict__.get("algorithm_registry")
        if not isinstance(registry, Mapping) or not registry:
            raise TypeError(
                f"{cls.__name__} must declare a non-empty "
                "``algorithm_registry: ClassVar[Mapping[str, ...]]`` ClassVar. "
                "Abstract intermediate classes can opt out via "
                "``class Foo(SemanticallyDescribable, register=False): ...``."
            )
