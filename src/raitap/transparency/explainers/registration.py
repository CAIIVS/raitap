"""Family decorator for transparency explainers.

Adapter sites use ``@register_transparency_adapter(...)`` instead of the
legacy ``class Foo(AttributionOnlyExplainer, registry_name=..., extra=..., ...)``
class-kwargs syntax. ``registry_name`` is pyright-checked at the decoration
site via ``_CommonRegKwargs.registry_name: Required[str]``. The family-required
class-body attributes (``output_payload_kind``, ``algorithm_registry``) are
validated at decoration time by :func:`_register_core` against
``TRANSPARENCY.required_classvars`` — see :class:`raitap._adapters.FamilyConfig`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import FamilyConfig, _CommonRegKwargs, _register_core
from raitap.configs.schema import TransparencyConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.transparency.explainers.base_explainer import BaseExplainer

TRANSPARENCY = FamilyConfig(
    group="transparency",
    schema=TransparencyConfig,
    package_style="nested",
    strip_suffixes=("Explainer",),
    has_algorithm_registry=True,
)

T = TypeVar("T", bound="BaseExplainer")


def register_transparency_adapter(
    **common: Unpack[_CommonRegKwargs],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a transparency explainer.

    Adapter class body must declare ``output_payload_kind`` and
    ``algorithm_registry`` (enforced at decoration time via
    ``TRANSPARENCY.required_classvars``). Adapters that ship ONNX-compatible
    algorithm subsets keep that as a class-body ``ONNX_COMPATIBLE_ALGORITHMS``
    frozenset.
    """

    def wrap(cls: type[T]) -> type[T]:
        return _register_core(cls, family=TRANSPARENCY, **common)

    return wrap
