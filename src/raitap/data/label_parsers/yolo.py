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

    def _build_image_index(self, image_dir: Path) -> dict[str, list[Path]]:
        # Walk the image tree ONCE (stem -> sorted image paths) so the fallback
        # lookup is O(1) per label instead of an rglob-per-label.
        suffixes = set(_IMAGE_SUFFIXES)
        index: dict[str, list[Path]] = {}
        for path in image_dir.rglob("*"):
            if path.suffix.lower() in suffixes:
                index.setdefault(path.stem, []).append(path)
        for paths in index.values():
            paths.sort()
        return index

    def _image_for(self, image_dir: Path, rel_stem: Path, index: dict[str, list[Path]]) -> Path:
        # ``rel_stem`` is the label path relative to ``labels_dir`` without its
        # suffix (e.g. ``train/a``). Prefer the mirrored image layout
        # (``image_dir/train/a.jpg``); fall back to the by-stem index so
        # flat-label / nested-image layouts still resolve.
        for suffix in _IMAGE_SUFFIXES:
            candidate = image_dir / f"{rel_stem}{suffix}"
            if candidate.exists():
                return candidate
        matches = index.get(rel_stem.name)
        if matches:
            if len(matches) > 1:
                found = [m.relative_to(image_dir).as_posix() for m in matches]
                raise ValueError(
                    f"YOLO label {rel_stem.as_posix()!r} has no mirrored image and its "
                    f"stem is ambiguous across multiple images {found}; use a mirrored "
                    "labels/images directory layout so the match is unambiguous."
                )
            return matches[0]
        raise ValueError(
            f"YOLO parser found no image for label {rel_stem.as_posix()!r} under {image_dir}."
        )

    def _to_detection_records(self, labels_dir: Path, image_dir: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        index = self._build_image_index(image_dir)
        txts = sorted(labels_dir.rglob("*.txt"), key=lambda p: p.relative_to(labels_dir).as_posix())
        for txt in txts:
            rel_stem = txt.relative_to(labels_dir).with_suffix("")
            image_path = self._image_for(image_dir, rel_stem, index)
            with Image.open(image_path) as im:
                width, height = im.size
            boxes: list[list[float]] = []
            labels: list[int] = []
            for line in txt.read_text(encoding="utf-8").splitlines():
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
            sample_id = image_path.relative_to(image_dir).as_posix()
            records.append({"sample_id": sample_id, "boxes": boxes, "labels": labels})
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
