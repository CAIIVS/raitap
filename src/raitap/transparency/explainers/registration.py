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
    AdapterDecoratorOptions,
    FamilyConfig,
    _register_core,
)
from raitap.configs.schema import TransparencyConfig
from raitap.transparency.contracts import ExplanationPayloadKind

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from raitap.transparency.contracts import ExplainerAlgorithmSpec
    from raitap.transparency.explainers.base_explainer import BaseExplainer

TRANSPARENCY = FamilyConfig(
    group="transparency",
    schema=TransparencyConfig,
    package_style="nested",
)

T = TypeVar("T", bound="BaseExplainer")


def transparency_adapter(
    *,
    algorithm_registry: Mapping[str, ExplainerAlgorithmSpec],
    output_payload_kind: ExplanationPayloadKind = ExplanationPayloadKind.ATTRIBUTIONS,
    baseline_kwarg_name: str | None = None,
    **common: Unpack[AdapterDecoratorOptions],
) -> Callable[[type[T]], type[T]]:
    """Decorator: register a transparency explainer.

    Required:
        ``registry_name`` (via ``AdapterDecoratorOptions.Required[str]``) and
        ``algorithm_registry`` (algorithm name → :class:`ExplainerAlgorithmSpec`).

    Optional:
        ``output_payload_kind`` defaults to ``ATTRIBUTIONS`` — the common case;
        override only if the explainer emits a different payload kind.
        ``baseline_kwarg_name`` is the call kwarg holding this family's reference
        input (``"baselines"`` for Captum, ``"background_data"`` for SHAP);
        ``None`` (default) means the family takes no baseline. The per-algorithm
        implicit default mode lives on each ``ExplainerAlgorithmSpec.baseline_default``.

        Per-algorithm backend needs live on each hint's ``requires`` (see raitap.types.Capability).
    """

    def wrap(cls: type[T]) -> type[T]:
        cls.algorithm_registry = algorithm_registry  # type: ignore[misc]
        cls.output_payload_kind = output_payload_kind
        cls.baseline_kwarg_name = baseline_kwarg_name
        return _register_core(cls, family=TRANSPARENCY, **common)

    return wrap
