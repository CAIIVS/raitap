"""Family decorator for robustness assessors.

Adapter sites use ``@register_robustness_adapter(...)`` instead of the
legacy ``class Foo(EmpiricalAttackAssessor, registry_name=..., extra=..., ...)``
class-kwargs syntax. ``registry_name`` is required (enforced via
``Required[str]`` in ``_CommonRegKwargs``) so pyright errors at the decoration
site if omitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import FamilyConfig, _CommonRegKwargs, _register_core
from raitap.configs.schema import RobustnessConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.robustness.assessors.base_assessor import BaseAssessor

ROBUSTNESS = FamilyConfig(
    group="robustness",
    schema=RobustnessConfig,
    package_style="nested",
    has_algorithm_registry=True,
)

T = TypeVar("T", bound="BaseAssessor")


def register_robustness_adapter(
    **common: Unpack[_CommonRegKwargs],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a robustness assessor.

    Adapter class body must declare ``algorithm_registry`` (enforced at
    decoration time via ``ROBUSTNESS.required_classvars``).
    """

    def wrap(cls: type[T]) -> type[T]:
        return _register_core(cls, family=ROBUSTNESS, **common)

    return wrap
