"""Base protocol and type alias for label parsers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from raitap.types import TaskKind  # noqa: TC001  must stay runtime-resolvable for get_type_hints()

# Documentation alias for the union of parsed label representations. Kept as a
# string (not a runtime annotation) so it needs no runtime ``torch`` import;
# annotations use ``Any`` since the concrete shape is task-family dependent.
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
