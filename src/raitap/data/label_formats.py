"""Pluggable label-format adapters (issue #338).

Each adapter converts an external annotation file (COCO / YOLO / VOC) into
RAITAP's native intermediate record list, which the task-family loaders then
align to ``sample_ids`` with their existing logic. Registry mirrors
``raitap.task_families.registry``: a decorator registers one singleton per
``LabelFormat``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.data.types import LabelFormat
    from raitap.types import TaskKind

#: Native intermediate record shapes (match the on-disk native formats).
DetectionRecord = dict[str, Any]
ClassificationRecord = dict[str, Any]


@runtime_checkable
class LabelFormatAdapter(Protocol):
    """Converts an external label file to native intermediate records."""

    format: LabelFormat
    supported_tasks: frozenset[TaskKind]

    def to_detection_records(
        self,
        source: Path,
        *,
        image_dir: Path | None,
        class_names: list[str] | None,
    ) -> list[DetectionRecord]:
        """Return ``[{sample_id, boxes (xyxy), labels}]``. Raise if unsupported."""
        ...

    def to_classification_records(self, source: Path) -> list[ClassificationRecord]:
        """Return ``[{sample_id, label}]``. Raise if unsupported."""
        ...


#: format -> the adapter singleton serving it.
LABEL_FORMAT_ADAPTERS: dict[LabelFormat, LabelFormatAdapter] = {}

T = TypeVar("T")


def label_format(cls: type[T]) -> type[T]:
    """Register ``cls`` (instantiated once) under its ``format`` class attribute."""
    instance = cls()  # type: ignore[call-arg]
    LABEL_FORMAT_ADAPTERS[instance.format] = instance  # type: ignore[attr-defined]
    return cls


def resolve_label_format_adapter(fmt: LabelFormat, *, task_kind: TaskKind) -> LabelFormatAdapter:
    """Return the adapter for ``fmt`` that supports ``task_kind``.

    Raises ``ValueError`` when no adapter is registered for ``fmt`` (e.g.
    ``native``, which the caller should special-case) or the adapter does not
    declare ``task_kind`` in ``supported_tasks``.
    """
    # Import side-effect: register the in-tree adapters on first use.
    from raitap.data import (
        _label_format_adapters,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    )

    adapter = LABEL_FORMAT_ADAPTERS.get(fmt)
    if adapter is None:
        raise ValueError(
            f"No adapter registered for label format {fmt.value!r}; "
            f"registered: {sorted(f.value for f in LABEL_FORMAT_ADAPTERS)}."
        )
    if task_kind not in adapter.supported_tasks:
        supported = sorted(t.value for t in adapter.supported_tasks)
        raise ValueError(
            f"Label format {fmt.value!r} does not support task {task_kind.value!r}; "
            f"supported tasks: {supported}."
        )
    return adapter
