"""COCO label-format adapter (issue #338)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from raitap.data.label_formats import (
    ClassificationRecord,
    DetectionRecord,
    label_format,
)
from raitap.data.types import LabelFormat
from raitap.types import TaskKind


@label_format
class CocoAdapter:
    """COCO ``instances.json`` -> native records.

    Detection: ``bbox`` is ``[x, y, w, h]`` -> ``[x1, y1, x2, y2]``;
    ``category_id`` passes through unchanged so labels stay in the model's
    label space. Classification: one label per image (the image's single
    annotation category); images with 0 or >1 categories raise.
    """

    format = LabelFormat.coco
    supported_tasks = frozenset({TaskKind.detection, TaskKind.classification})

    def _load(self, source: Path) -> dict[str, Any]:
        with source.open() as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "images" not in data:
            raise ValueError(
                f"COCO file {source} must be an object with an 'images' array."
            )
        return data

    def to_detection_records(
        self, source: Path, *, image_dir: Path | None, class_names: list[str] | None
    ) -> list[DetectionRecord]:
        data = self._load(source)
        file_by_image: dict[int, str] = {
            img["id"]: img["file_name"] for img in data["images"]
        }
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

    def to_classification_records(
        self, source: Path
    ) -> list[ClassificationRecord]:
        data = self._load(source)
        file_by_image: dict[int, str] = {
            img["id"]: img["file_name"] for img in data["images"]
        }
        cats: dict[int, set[int]] = {iid: set() for iid in file_by_image}
        for ann in data.get("annotations", []):
            cats[ann["image_id"]].add(int(ann["category_id"]))
        records: list[ClassificationRecord] = []
        for iid, name in file_by_image.items():
            cat_set = cats[iid]
            if len(cat_set) != 1:
                raise ValueError(
                    f"COCO classification needs exactly one category per image; "
                    f"image {name!r} has {len(cat_set)}."
                )
            records.append({"sample_id": name, "label": next(iter(cat_set))})
        return records
