"""Family decorator for transparency explainers.

Adapter sites use ``@register_transparency_adapter(...)`` instead of the
legacy ``class Foo(AttributionOnlyExplainer, registry_name=..., extra=..., ...)``
class-kwargs syntax. Required per-family metadata (``algorithm_registry``) is
type-checked by pyright at the decoration site. ``output_payload_kind``
defaults to ``ATTRIBUTIONS`` — the common case — but can be overridden per
adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import FamilyConfig, _CommonRegKwargs, _register_core
from raitap.configs.schema import TransparencyConfig
from raitap.transparency.contracts import ExplanationPayloadKind

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from raitap.transparency.contracts import MethodFamily
    from raitap.transparency.explainers.base_explainer import BaseExplainer

TRANSPARENCY = FamilyConfig(
    group="transparency",
    schema=TransparencyConfig,
    package_style="nested",
)

T = TypeVar("T", bound="BaseExplainer")


def register_transparency_adapter(
    *,
    algorithm_registry: Mapping[str, frozenset[MethodFamily]],
    output_payload_kind: ExplanationPayloadKind = ExplanationPayloadKind.ATTRIBUTIONS,
    onnx_compatible_algorithms: frozenset[str] = frozenset(),
    **common: Unpack[_CommonRegKwargs],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a transparency explainer.

    Required:
        ``registry_name`` (via ``_CommonRegKwargs.Required[str]``) and
        ``algorithm_registry``.

    Optional:
        ``output_payload_kind`` defaults to ``ATTRIBUTIONS`` (most explainers
        emit attributions). ``onnx_compatible_algorithms`` defaults to empty.
    """

    def wrap(cls: type[T]) -> type[T]:
        cls.algorithm_registry = algorithm_registry  # type: ignore[misc]
        cls.output_payload_kind = output_payload_kind
        cls.ONNX_COMPATIBLE_ALGORITHMS = onnx_compatible_algorithms  # type: ignore[attr-defined]
        return _register_core(cls, family=TRANSPARENCY, **common)

    return wrap
