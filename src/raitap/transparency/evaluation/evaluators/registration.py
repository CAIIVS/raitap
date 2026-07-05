"""Family-less decorator for transparency explanation-quality evaluators (#341).

Registers ``family=None`` (like visualisers): no hydra config group, instantiated
via ``_target_`` nested under ``transparency.<name>.evaluation``. ``extra`` and
``library`` MUST be passed explicitly (no family default).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import AdapterDecoratorOptions, _register_core

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from raitap.transparency.evaluation.contracts import QuantusMetricSpec
    from raitap.transparency.evaluation.evaluators.base_evaluator import BaseEvaluator

T = TypeVar("T", bound="BaseEvaluator")


def transparency_evaluator(
    *,
    algorithm_registry: Mapping[str, QuantusMetricSpec],
    **common: Unpack[AdapterDecoratorOptions],
) -> Callable[[type[T]], type[T]]:
    def wrap(cls: type[T]) -> type[T]:
        cls.algorithm_registry = algorithm_registry  # type: ignore[misc]
        return _register_core(cls, family=None, **common)

    return wrap
