"""Family decorator for transparency explainers.

Adapter sites use ``@register_transparency_adapter(...)`` instead of the
legacy ``class Foo(AttributionOnlyExplainer, registry_name=..., extra=..., ...)``
class-kwargs syntax. Required per-family metadata (``output_payload_kind``,
``algorithm_registry``) is type-checked by pyright at the decoration site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import FamilyConfig, _CommonRegKwargs, _register_core
from raitap.configs.schema import TransparencyConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from raitap.transparency.contracts import ExplanationPayloadKind, MethodFamily
    from raitap.transparency.explainers.base_explainer import BaseExplainer

TRANSPARENCY = FamilyConfig(
    group="transparency",
    schema=TransparencyConfig,
    package_style="nested",
    strip_suffixes=("Explainer",),
)

T = TypeVar("T", bound="BaseExplainer")


def register_transparency_adapter(
    *,
    output_payload_kind: "ExplanationPayloadKind",
    algorithm_registry: "Mapping[str, frozenset[MethodFamily]]",
    onnx_compatible_algorithms: frozenset[str] = frozenset(),
    **common: Unpack[_CommonRegKwargs],
) -> "Callable[[type[T]], type[T]]":
    """Decorator: register a transparency explainer.

    ``registry_name`` is required (enforced via ``Required[str]`` in
    ``_CommonRegKwargs``). ``output_payload_kind`` and ``algorithm_registry``
    are transparency-specific required kwargs — pyright errors at the call
    site if either is missing.
    """

    def wrap(cls: type[T]) -> type[T]:
        cls.output_payload_kind = output_payload_kind  # type: ignore[misc]
        cls.algorithm_registry = algorithm_registry  # type: ignore[misc]
        cls.ONNX_COMPATIBLE_ALGORITHMS = onnx_compatible_algorithms  # type: ignore[misc]
        return _register_core(cls, family=TRANSPARENCY, **common)

    return wrap
