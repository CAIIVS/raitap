"""Validate inferred extras against pyproject's ``[tool.uv].conflicts``.

Each conflict group is a list of single-key ``{extra = "name"}`` tables. A
chosen set must hit at most one extra per group; the validator raises
:class:`ExtrasConflictError` with the conflicting names and their inference
origins when the check fails.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path


class ExtrasConflictError(RuntimeError):
    """Raised when the inferred extras set violates a uv conflict group."""


def load_conflict_groups(pyproject_path: Path) -> list[frozenset[str]]:
    """Return conflict groups declared under ``[tool.uv].conflicts``.

    Missing table or empty list yields ``[]``.
    """
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    raw = data.get("tool", {}).get("uv", {}).get("conflicts", [])
    groups: list[frozenset[str]] = []
    for entry in raw:
        members: set[str] = set()
        for member in entry:
            name = member.get("extra")
            if isinstance(name, str):
                members.add(name)
        if members:
            groups.append(frozenset(members))
    return groups


def validate_conflicts(
    extras: Iterable[str],
    pyproject_path: Path,
    *,
    origins: Mapping[str, str],
) -> None:
    """Raise :class:`ExtrasConflictError` if ``extras`` violates any group.

    ``origins`` maps each extra name to a short human-readable phrase
    describing why it was selected. Used to build a precise error message.
    """
    selected = set(extras)
    for group in load_conflict_groups(pyproject_path):
        clash = selected & group
        if len(clash) <= 1:
            continue
        parts = [
            f"{name} ({origins.get(name, 'no origin recorded')})"
            for name in sorted(clash)
        ]
        raise ExtrasConflictError(
            "Inferred extras violate a pyproject conflict group: " + ", ".join(parts)
        )
