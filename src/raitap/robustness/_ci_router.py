"""Resolve raitap-namespaced CI settings for average-case assessors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

if TYPE_CHECKING:
    from .assessors._ci import CIMethod


def resolve_ci_settings(raitap_kwargs: dict[str, Any]) -> tuple[CIMethod, float]:
    from .assessors._ci import CIMethod

    method = raitap_kwargs.pop("ci_method", "wilson")
    valid = get_args(CIMethod)
    if method not in valid:
        allowed = " or ".join(repr(m) for m in valid)
        raise ValueError(f"raitap.ci_method must be {allowed}, got {method!r}.")
    level = raitap_kwargs.pop("ci_level", 0.95)
    if not isinstance(level, (int, float)) or not 0.0 < float(level) < 1.0:
        raise ValueError(f"raitap.ci_level must be in (0, 1), got {level!r}.")
    return method, float(level)
