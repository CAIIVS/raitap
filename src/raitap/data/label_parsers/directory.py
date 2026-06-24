"""Directory label parser stub (real logic lands in Task 3)."""

from __future__ import annotations

from typing import Any

from raitap.configs.schema import DirectoryLabelsConfig
from raitap.data.label_parsers.registration import label_parser
from raitap.types import TaskKind


@label_parser(registry_name="directory", schema=DirectoryLabelsConfig)
class DirectoryLabelParser:
    """Parse labels from directory structure (stub; returns None until Task 3)."""

    supported_tasks: frozenset[TaskKind] = frozenset({TaskKind.classification})

    def parse(
        self,
        *,
        task_kind: TaskKind,
        tensor: Any,
        sample_ids: list[str] | None,
        data_source: str | None,
        class_names: list[str] | None,
    ) -> None:
        return None
