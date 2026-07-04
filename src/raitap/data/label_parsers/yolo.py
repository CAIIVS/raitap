"""YOLO label parser (detection-only)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image

from raitap.configs.schema import YoloLabelsConfig
from raitap.data.data import SourceKind, get_source_path
from raitap.data.label_parsers.registration import label_parser
from raitap.data.types import IdStrategy
from raitap.task_families.detection import _align_detection_records
from raitap.types import TaskKind

if TYPE_CHECKING:
    from pathlib import Path

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@label_parser(registry_name="yolo", schema=YoloLabelsConfig)
class YoloLabelParser:
    """Parse YOLO per-image ``.txt`` (``class cx cy w h``, normalised) for detection.

    Boxes are denormalised to pixel ``[x1, y1, x2, y2]`` using each image's
    size read from PIL. Class indices pass through unchanged.
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

    def _image_for(self, image_dir: Path, stem: str) -> Path:
        for suffix in _IMAGE_SUFFIXES:
            candidate = image_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        raise ValueError(f"YOLO parser found no image for label {stem!r} in {image_dir}.")

    def _to_detection_records(self, labels_dir: Path, image_dir: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for txt in sorted(labels_dir.glob("*.txt")):
            image_path = self._image_for(image_dir, txt.stem)
            with Image.open(image_path) as im:
                width, height = im.size
            boxes: list[list[float]] = []
            labels: list[int] = []
            for line in txt.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                if len(parts) < 5:
                    raise ValueError(
                        f"YOLO label {txt.name} has a line with {len(parts)} "
                        f"field(s), expected 5 (class cx cy w h): {line!r}."
                    )
                cls, cx, cy, bw, bh = (float(p) for p in parts[:5])
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))
            records.append({"sample_id": image_path.name, "boxes": boxes, "labels": labels})
        return records

    def parse(
        self,
        *,
        task_kind: TaskKind,
        tensor: Any,
        sample_ids: list[str] | None,
        data_source: str | None,
        class_names: list[str] | None,
    ) -> Any:
        """Load YOLO labels and align to sample_ids for detection."""
        if data_source is None:
            raise ValueError(
                "YOLO labels need data.source (image directory) to denormalise boxes; "
                "set data.source to the image directory."
            )
        labels_dir = get_source_path(self.source, kind=SourceKind.LABELS)
        image_dir = get_source_path(data_source, kind=SourceKind.DATA)
        records = self._to_detection_records(labels_dir, image_dir)
        return _align_detection_records(
            records,
            expected=len(tensor),
            sample_ids=sample_ids,
            strategy=str(self.id_strategy),
        )
