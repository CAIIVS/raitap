"""Family decorator for robustness assessors.

Adapter sites use ``@register_robustness_adapter(...)`` instead of the legacy
``class Foo(EmpiricalAttackAssessor, registry_name=..., extra=..., ...)``
class-kwargs syntax. ``registry_name`` + ``algorithm_registry`` are
pyright-checked at the decoration site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import FamilyConfig, _CommonRegKwargs, _register_core
from raitap.configs.schema import RobustnessConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from raitap.robustness.assessors.base_assessor import BaseAssessor
    from raitap.robustness.semantics import AssessorSemanticsHints

ROBUSTNESS = FamilyConfig(
    group="robustness",
    schema=RobustnessConfig,
    package_style="nested",
)

T = TypeVar("T", bound="BaseAssessor")


def register_robustness_adapter(
    *,
    algorithm_registry: Mapping[str, AssessorSemanticsHints],
    **common: Unpack[_CommonRegKwargs],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a robustness assessor.

    Required: ``registry_name`` and ``algorithm_registry`` (algorithms being a
    core RAITAP concept for both transparency and robustness adapters).
    """

    def wrap(cls: type[T]) -> type[T]:
        cls.algorithm_registry = algorithm_registry  # type: ignore[misc]
        return _register_core(cls, family=ROBUSTNESS, **common)

    return wrap
