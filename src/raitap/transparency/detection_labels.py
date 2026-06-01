"""Pure helpers for detection box labelling (issue #233).

No I/O, no model, no GPU — keeps id->name resolution and box->GT matching
unit-testable on synthetic tensors. ``resolve_category_names`` fixes source
precedence (explicit config > backend weights.meta > None) in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def resolve_category_names(
    explicit: Sequence[str] | None,
    backend_categories: Sequence[str] | None,
) -> list[str] | None:
    """Resolve the id->name table. Precedence: explicit config > backend > None."""
    if explicit is not None:
        return list(explicit)
    if backend_categories is not None:
        return list(backend_categories)
    return None


def label_name_for(index: int, names: Sequence[str] | None) -> str | None:
    """Bounds-safe id->name lookup; ``None`` when no map or index out of range."""
    if names is None:
        return None
    if 0 <= index < len(names):
        return names[index]
    return None
