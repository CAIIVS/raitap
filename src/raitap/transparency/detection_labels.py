"""Pure helpers for detection box labelling (issue #233).

No I/O, no model, no GPU — keeps id->name resolution and box->GT matching
unit-testable on synthetic tensors. ``resolve_category_names`` fixes source
precedence (explicit config > backend weights.meta > None) in one place.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from raitap.transparency.contracts import DetectionBox


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


def enrich_detection_box(
    box: DetectionBox,
    *,
    category_names: Sequence[str] | None = None,
) -> DetectionBox:
    """Return *box* with display metadata resolved from the category-names table.

    Pure: no model, no I/O. ``explain_detection`` builds the raw box (prediction
    geometry + ``label_index``); the transparency caller calls this to fill the
    human-readable ``label_name`` before rendering. A later task extends this
    helper with the ground-truth match (``gt_*`` params + fields).
    """
    return dataclasses.replace(
        box,
        label_name=label_name_for(box.label_index, category_names),
    )
