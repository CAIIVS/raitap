"""Resolve raitap-namespaced CI settings for average-case assessors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .assessors._ci import CIMethod


def resolve_ci_settings(raitap_kwargs: dict[str, Any]) -> tuple[CIMethod, float]:
    method = raitap_kwargs.pop("ci_method", "wilson")
    if method not in ("wilson", "clopper_pearson"):
        raise ValueError(f"raitap.ci_method must be 'wilson' or 'clopper_pearson', got {method!r}.")
    level = raitap_kwargs.pop("ci_level", 0.95)
    if not isinstance(level, (int, float)) or not 0.0 < float(level) < 1.0:
        raise ValueError(f"raitap.ci_level must be in (0, 1), got {level!r}.")
    return method, float(level)
