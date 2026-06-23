"""YOLO label-format adapter (issue #338)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from raitap.data.label_formats import (
    ClassificationRecord,
    DetectionRecord,
    label_format,
)
from raitap.data.types import LabelFormat
from raitap.types import TaskKind

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@label_format
class YoloAdapter:
    """YOLO per-image ``.txt`` (``class cx cy w h``, normalised) -> native records.

    Boxes are denormalised with each image's pixel size, read from
    ``image_dir``. Class indices pass through unchanged.
    """

    format = LabelFormat.yolo
    supported_tasks = frozenset({TaskKind.detection})

    def _image_for(self, image_dir: Path, stem: str) -> Path:
        for suffix in _IMAGE_SUFFIXES:
            candidate = image_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        raise ValueError(
            f"YOLO adapter found no image for label {stem!r} in {image_dir}."
        )

    def to_detection_records(
        self, source: Path, *, image_dir: Path | None, class_names: list[str] | None
    ) -> list[DetectionRecord]:
        if image_dir is None:
            raise ValueError(
                "YOLO labels need image_dir to denormalise boxes; "
                "set data.source to the image directory."
            )
        records: list[DetectionRecord] = []
        for txt in sorted(source.glob("*.txt")):
            image_path = self._image_for(image_dir, txt.stem)
            with Image.open(image_path) as im:
                width, height = im.size
            boxes: list[list[float]] = []
            labels: list[int] = []
            for line in txt.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls, cx, cy, bw, bh = (float(p) for p in parts[:5])
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))
            records.append(
                {"sample_id": image_path.name, "boxes": boxes, "labels": labels}
            )
        return records

    def to_classification_records(
        self, source: Path
    ) -> list[ClassificationRecord]:
        raise ValueError("YOLO is a detection-only format.")
