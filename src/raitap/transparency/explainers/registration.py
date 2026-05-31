"""Family decorator for transparency explainers.

Adapter sites use ``@adapters.transparency(...)`` instead of the
legacy ``class Foo(AttributionOnlyExplainer, registry_name=..., extra=..., ...)``
class-kwargs syntax. Required per-family metadata (``algorithm_registry``) is
type-checked by pyright at the decoration site. ``output_payload_kind``
defaults to ``ATTRIBUTIONS`` — the common case — but can be overridden per
adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

from raitap._adapters import (
    ALL,
    AdapterDecoratorOptions,
    FamilyConfig,
    _AllAlgorithmsSentinel,
    _register_core,
)
from raitap.configs.schema import TransparencyConfig
from raitap.transparency.contracts import ExplanationPayloadKind

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from raitap.transparency.contracts import ExplainerSemanticsHints
    from raitap.transparency.explainers.base_explainer import BaseExplainer

TRANSPARENCY = FamilyConfig(
    group="transparency",
    schema=TransparencyConfig,
    package_style="nested",
)

T = TypeVar("T", bound="BaseExplainer")


def transparency_adapter(
    *,
    algorithm_registry: Mapping[str, ExplainerSemanticsHints],
    output_payload_kind: ExplanationPayloadKind = ExplanationPayloadKind.ATTRIBUTIONS,
    onnx_compatible_algorithms: frozenset[str] | _AllAlgorithmsSentinel = frozenset(),
    baseline_kwarg: str | None = None,
    **common: Unpack[AdapterDecoratorOptions],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a transparency explainer.

    Required:
        ``registry_name`` (via ``AdapterDecoratorOptions.Required[str]``) and
        ``algorithm_registry`` (algorithm name → :class:`ExplainerSemanticsHints`).

    Optional:
        ``output_payload_kind`` defaults to ``ATTRIBUTIONS`` — the common case;
        override only if the explainer emits a different payload kind.
        ``onnx_compatible_algorithms`` defaults to "none" (ONNX support is
        rare). Pass an explicit ``frozenset({"name1", "name2"})`` to enable a
        subset, or :data:`raitap.transparency.ALL` to enable every algorithm in
        ``algorithm_registry``.
        ``baseline_kwarg`` is the call kwarg holding this family's reference
        input (``"baselines"`` for Captum, ``"background_data"`` for SHAP);
        ``None`` (default) means the family takes no baseline. The per-algorithm
        implicit default mode lives on each ``ExplainerSemanticsHints.baseline_default``.
    """

    def wrap(cls: type[T]) -> type[T]:
        cls.algorithm_registry = algorithm_registry  # type: ignore[misc]
        cls.output_payload_kind = output_payload_kind
        cls.baseline_kwarg = baseline_kwarg
        cls.ONNX_COMPATIBLE_ALGORITHMS = (  # type: ignore[misc]
            frozenset(algorithm_registry.keys())
            if onnx_compatible_algorithms is ALL
            else onnx_compatible_algorithms
        )
        return _register_core(cls, family=TRANSPARENCY, **common)

    return wrap
