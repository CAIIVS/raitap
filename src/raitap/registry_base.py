"""Cross-module base interface for adapters that publish an algorithm registry.

Both ``raitap.transparency`` and ``raitap.robustness`` describe their adapters
by mapping algorithm-name → typed semantics payload (``frozenset[MethodFamily]``
for explainers, ``AssessorSemanticsHints`` for assessors). This module provides
the single shared contract:

* ``WithAlgorithmRegistry[T]`` — generic ABC; subclasses declare
  ``algorithm_registry: ClassVar[Mapping[str, T]]`` as a non-empty mapping.
* ``__init_subclass__`` enforces the registry at class-definition time so
  configuration mistakes fail at import, not at runtime mid-pipeline.

Intermediate abstract base classes (e.g. ``BaseAssessor``,
``EmpiricalAttackAssessor``, ``BaseExplainer``) opt out of validation by
passing ``abstract=True`` in their class signature; concrete adapters always
opt in (default ``abstract=False``).
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Any, ClassVar, Generic, TypeVar

T = TypeVar("T")


class WithAlgorithmRegistry(ABC, Generic[T]):
    """Adapter that publishes an algorithm-name → hints registry as a ClassVar.

    Subclasses must declare a non-empty ``algorithm_registry`` ClassVar at
    class-definition time. Pass ``abstract=True`` on the class line for
    abstract intermediate classes that don't (yet) carry concrete algorithms.

    The ``Generic[T]`` parameter is enforced statically only — pyright requires
    a ``# type: ignore[misc]`` on the base ClassVar declaration because PEP
    526 forbids ``ClassVar`` parameterised by a TypeVar. Concrete subclasses
    annotate ``algorithm_registry`` with the resolved type and pyright checks
    them normally.
    """

    algorithm_registry: ClassVar[Mapping[str, Any]]  # type: ignore[misc]
    """Concrete subclasses narrow the value type to ``Mapping[str, T]``."""

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # ``raitap._adapters.AdapterMixin`` detects abstractness independently
        # via ``inspect.isabstract``, so ``abstract`` only needs to gate this
        # mixin's own ``algorithm_registry`` validation.
        if abstract:
            return
        registry = cls.__dict__.get("algorithm_registry")
        if not isinstance(registry, Mapping) or not registry:
            raise TypeError(
                f"{cls.__name__} must declare a non-empty "
                "``algorithm_registry: ClassVar[Mapping[str, ...]]`` ClassVar. "
                "Abstract intermediate classes can opt out via "
                "``class Foo(WithAlgorithmRegistry, abstract=True): ...``."
            )
