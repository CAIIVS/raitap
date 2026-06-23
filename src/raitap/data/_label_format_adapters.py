# pyright: reportUnusedImport=false
"""Imports every in-tree label-format adapter so the decorators fire.

Imported for its side effects by
``raitap.data.label_formats.resolve_label_format_adapter``. Every import in this
module is intentionally side-effect-only (registers an adapter), so the
file-level ``reportUnusedImport=false`` above is correct.
"""

from __future__ import annotations

from raitap.data.adapters import coco, voc, yolo  # noqa: F401
