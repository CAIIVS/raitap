"""COCO label parser (detection + classification)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from raitap.configs.schema import CocoLabelsConfig
from raitap.data.data import (
    SourceKind,
    _align_labels_to_samples,
    _resolve_id_strategy,
    get_source_path,
)
from raitap.data.label_parsers.registration import label_parser
from raitap.data.types import IdStrategy
from raitap.task_families.detection import _align_detection_records
from raitap.types import TaskKind

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


@label_parser(registry_name="coco", schema=CocoLabelsConfig)
class CocoLabelParser:
    """Parse COCO ``instances.json`` labels for detection or classification.

    Detection: ``bbox`` ``[x, y, w, h]`` -> ``[x1, y1, x2, y2]``; ``category_id``
    passes through unchanged. Classification: one category per image; images with
    0 or >1 categories raise ValueError.
    """

    supported_tasks: frozenset[TaskKind] = frozenset({TaskKind.detection, TaskKind.classification})

    def __init__(
        self,
        *,
        source: str,
        id_strategy: IdStrategy = IdStrategy.auto,
    ) -> None:
        self.source = source
        self.id_strategy = id_strategy

    # --- internal helpers (ported verbatim from adapters/coco.py) ---

    def _load(self, source: Path) -> dict[str, Any]:
        with source.open() as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "images" not in data:
            raise ValueError(f"COCO file {source} must be an object with an 'images' array.")
        return data

    def _to_detection_records(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        file_by_image: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
        boxes: dict[int, list[list[float]]] = {iid: [] for iid in file_by_image}
        labels: dict[int, list[int]] = {iid: [] for iid in file_by_image}
        for ann in data.get("annotations", []):
            iid = ann["image_id"]
            x, y, w, h = ann["bbox"]
            boxes[iid].append([x, y, x + w, y + h])
            labels[iid].append(int(ann["category_id"]))
        return [
            {"sample_id": file_by_image[iid], "boxes": boxes[iid], "labels": labels[iid]}
            for iid in file_by_image
        ]

    def _to_classification_records(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        file_by_image: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
        cats: dict[int, set[int]] = {iid: set() for iid in file_by_image}
        for ann in data.get("annotations", []):
            cats[ann["image_id"]].add(int(ann["category_id"]))
        records: list[dict[str, Any]] = []
        for iid, name in file_by_image.items():
            cat_set = cats[iid]
            if len(cat_set) != 1:
                raise ValueError(
                    f"COCO classification needs exactly one category per image; "
                    f"image {name!r} has {len(cat_set)}."
                )
            records.append({"sample_id": name, "label": next(iter(cat_set))})
        return records

    # --- public parse method ---

    def parse(
        self,
        *,
        task_kind: TaskKind,
        tensor: Any,
        sample_ids: list[str] | None,
        data_source: str | None,
        class_names: list[str] | None,
    ) -> Any:
        """Load and align COCO labels for detection or classification."""
        import pandas as pd

        labels_path = get_source_path(self.source, kind=SourceKind.LABELS)
        data = self._load(labels_path)

        if task_kind is TaskKind.detection:
            records = self._to_detection_records(data)
            return _align_detection_records(
                records,
                expected=len(tensor),
                sample_ids=sample_ids,
                strategy=str(self.id_strategy),
            )

        # classification
        records = self._to_classification_records(data)
        raw_ids: list[str] = [r["sample_id"] for r in records]
        encoded: list[int] = [r["label"] for r in records]
        id_series: pd.Series = pd.Series(raw_ids)
        strategy = _resolve_id_strategy(str(self.id_strategy), id_series)
        aligned = _align_labels_to_samples(
            sample_ids=sample_ids or [],
            raw_label_ids=id_series,
            encoded_labels=encoded,
            strategy=strategy,
        )
        import torch

        return torch.tensor(aligned, dtype=torch.long)
