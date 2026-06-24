"""Base protocol and type alias for label parsers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from raitap.types import TaskKind

# Type alias for the union of parsed label representations.
ParsedLabels = "torch.Tensor | list[dict[str, torch.Tensor]] | None"


@runtime_checkable
class LabelParser(Protocol):
    """Protocol every label-parser adapter must satisfy."""

    supported_tasks: frozenset[TaskKind]

    def parse(
        self,
        *,
        task_kind: TaskKind,
        tensor: Any,
        sample_ids: list[str] | None,
        data_source: str | None,
        class_names: list[str] | None,
    ) -> Any: ...
