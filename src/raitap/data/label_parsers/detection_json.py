"""Detection-JSON label parser (native RAITAP detection record format)."""

from __future__ import annotations

import json
from typing import Any

from raitap.configs.schema import DetectionJsonLabelsConfig
from raitap.data.data import SourceKind, get_source_path
from raitap.data.label_parsers.registration import label_parser
from raitap.data.types import IdStrategy
from raitap.task_families.detection import _align_detection_records
from raitap.types import TaskKind


@label_parser(registry_name="detection_json", schema=DetectionJsonLabelsConfig)
class DetectionJsonLabelParser:
    """Parse native RAITAP detection JSON records for detection.

    The file must be a JSON array of objects with keys ``sample_id``,
    ``boxes`` (list of ``[x1, y1, x2, y2]`` in pixels), and ``labels``
    (list of integer class ids).
    """

    supported_tasks: frozenset[TaskKind] = frozenset({TaskKind.detection})

    def __init__(
        self,
        *,
        source: str,
        id_strategy: IdStrategy = IdStrategy.auto,
    ) -> None:
        self.source = source
        self.id_strategy = id_strategy

    def parse(
        self,
        *,
        task_kind: TaskKind,
        tensor: Any,
        sample_ids: list[str] | None,
        data_source: str | None,
        class_names: list[str] | None,
    ) -> Any:
        """Load native detection JSON and align records to sample_ids."""
        labels_path = get_source_path(self.source, kind=SourceKind.LABELS)
        with labels_path.open() as fh:
            records = json.load(fh)
        if not isinstance(records, list):
            raise ValueError(f"Detection labels file {labels_path} must be a JSON array.")
        return _align_detection_records(
            records,
            expected=len(tensor),
            sample_ids=sample_ids,
            strategy=str(self.id_strategy),
        )
