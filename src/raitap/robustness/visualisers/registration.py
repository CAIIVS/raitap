"""Family decorator for robustness visualisers (no Hydra group). Capability
fields use the UNSET sentinel so omitted kwargs keep the base default."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypeVar, Unpack

from raitap._adapters import AdapterDecoratorOptions, _register_core

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.robustness.contracts import AssessmentKind
    from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser


class _Unset:
    __slots__ = ()


_UNSET: Final = _Unset()

T = TypeVar("T", bound="BaseRobustnessVisualiser")


def robustness_visualiser(
    *,
    supported_assessment_kinds: frozenset[AssessmentKind] | _Unset = _UNSET,
    embeds_clean_input: bool | _Unset = _UNSET,
    embeds_perturbation_map: bool | _Unset = _UNSET,
    **common: Unpack[AdapterDecoratorOptions],
) -> Callable[[type[T]], type[T]]:
    """Register a robustness visualiser. ``registry_name`` required; capability
    kwargs optional (omitted → base default)."""

    def wrap(cls: type[T]) -> type[T]:
        for attr, value in (
            ("supported_assessment_kinds", supported_assessment_kinds),
            ("embeds_clean_input", embeds_clean_input),
            ("embeds_perturbation_map", embeds_perturbation_map),
        ):
            if value is not _UNSET:  # identity check on the singleton sentinel
                setattr(cls, attr, value)
        return _register_core(cls, family=None, **common)

    return wrap
