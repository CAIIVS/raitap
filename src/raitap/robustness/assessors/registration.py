"""Family decorator for robustness assessors.

Adapter sites use ``@adapters.robustness(...)`` instead of the legacy
``class Foo(EmpiricalAttackAssessor, registry_name=..., extra=..., ...)``
class-kwargs syntax. ``registry_name`` + ``algorithm_registry`` are
pyright-checked at the decoration site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar, Unpack

from raitap._adapters import (
    ALL,
    AdapterDecoratorOptions,
    FamilyConfig,
    _AllAlgorithmsSentinel,
    _register_core,
)
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


def robustness_adapter(
    *,
    algorithm_registry: Mapping[str, AssessorSemanticsHints],
    onnx_compatible_algorithms: frozenset[str] | _AllAlgorithmsSentinel = frozenset(),
    budget_kwarg_source: Literal["init_kwargs", "call_kwargs"] = "init_kwargs",
    **common: Unpack[AdapterDecoratorOptions],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a robustness assessor.

    Required:
        ``registry_name`` and ``algorithm_registry`` (algorithms being a core
        RAITAP concept for both transparency and robustness adapters).

    Optional:
        ``onnx_compatible_algorithms`` defaults to "none" (most attack
        libraries require torch autograd, not ONNX). Pass an explicit
        ``frozenset({"name1", "name2"})`` or :data:`raitap.robustness.ALL`
        to enable all algorithms on ONNX backends.

        ``budget_kwarg_source`` controls where semantics looks for the
        perturbation budget: ``"init_kwargs"`` (default, budget passed at
        construction time) or ``"call_kwargs"`` (budget passed at attack-call
        time, e.g. foolbox-style ``epsilons=`` argument).
    """

    def wrap(cls: type[T]) -> type[T]:
        cls.algorithm_registry = algorithm_registry  # type: ignore[misc]
        cls.budget_kwarg_source = budget_kwarg_source  # type: ignore[misc]
        cls.ONNX_COMPATIBLE_ALGORITHMS = (  # type: ignore[misc]
            frozenset(algorithm_registry.keys())
            if onnx_compatible_algorithms is ALL
            else onnx_compatible_algorithms
        )
        return _register_core(cls, family=ROBUSTNESS, **common)

    return wrap
