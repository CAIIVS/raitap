"""Family decorator for transparency visualisers (no Hydra group).

Capability fields are passed as kwargs and set on the class only when provided
(UNSET sentinel), so values inherited from an intermediate base (e.g.
``_TabularSummaryContractMixin``) are preserved — behaviour stays identical to
the previous class-body declarations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypeVar, Unpack

from raitap._adapters import AdapterDecoratorOptions, _register_core

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.transparency.contracts import (
        ExplanationOutputSpace,
        ExplanationPayloadKind,
        ExplanationScope,
        MethodFamily,
        ScopeDefinitionStep,
        VisualSummarySpec,
    )
    from raitap.transparency.visualisers.base_visualiser import BaseVisualiser
    from raitap.types import TaskKind


class _Unset:
    __slots__ = ()


_UNSET: Final = _Unset()

T = TypeVar("T", bound="BaseVisualiser")


def transparency_visualiser(
    *,
    supported_payload_kinds: frozenset[ExplanationPayloadKind] | _Unset = _UNSET,
    supported_scopes: frozenset[ExplanationScope] | _Unset = _UNSET,
    supported_output_spaces: frozenset[ExplanationOutputSpace] | _Unset = _UNSET,
    supported_method_families: frozenset[MethodFamily] | _Unset = _UNSET,
    supported_tasks: frozenset[TaskKind] | _Unset = _UNSET,
    compatible_algorithms: frozenset[str] | _Unset = _UNSET,
    embeds_original_input: bool | _Unset = _UNSET,
    produces_scope: ExplanationScope | None | _Unset = _UNSET,
    scope_definition_step: ScopeDefinitionStep | None | _Unset = _UNSET,
    visual_summary: VisualSummarySpec | None | _Unset = _UNSET,
    **common: Unpack[AdapterDecoratorOptions],
) -> Callable[[type[T]], type[T]]:
    """Register a transparency visualiser. ``registry_name`` required; all
    capability kwargs optional (omitted → inherit base/mixin default)."""

    def wrap(cls: type[T]) -> type[T]:
        for attr, value in (
            ("supported_payload_kinds", supported_payload_kinds),
            ("supported_scopes", supported_scopes),
            ("supported_output_spaces", supported_output_spaces),
            ("supported_method_families", supported_method_families),
            ("supported_tasks", supported_tasks),
            ("compatible_algorithms", compatible_algorithms),
            ("embeds_original_input", embeds_original_input),
            ("produces_scope", produces_scope),
            ("scope_definition_step", scope_definition_step),
            ("visual_summary", visual_summary),
        ):
            if value is not _UNSET:  # identity check on the singleton sentinel
                setattr(cls, attr, value)
        return _register_core(cls, family=None, **common)

    return wrap
