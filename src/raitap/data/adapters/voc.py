"""Pascal-VOC label-format adapter (issue #338)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from raitap.data.label_formats import (
    ClassificationRecord,
    DetectionRecord,
    label_format,
)
from raitap.data.types import LabelFormat
from raitap.types import TaskKind

#: Canonical Pascal-VOC class order (index = label id) when no class_names given.
_VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def _coord(box: ET.Element, tag: str, xml_path: Path) -> float:
    text = box.findtext(tag)
    if text is None:
        raise ValueError(f"VOC bndbox in {xml_path.name} missing <{tag}>.")
    return float(text)


@label_format
class VocAdapter:
    """Pascal-VOC per-image ``.xml`` -> native detection records.

    Boxes are already ``[xmin, ymin, xmax, ymax]`` pixels. Class names map to
    ids by their position in ``class_names`` (else the standard 20-class VOC
    order).
    """

    format = LabelFormat.voc
    supported_tasks = frozenset({TaskKind.detection})

    def to_detection_records(
        self, source: Path, *, image_dir: Path | None, class_names: list[str] | None
    ) -> list[DetectionRecord]:
        name_to_id = {
            name: idx for idx, name in enumerate(class_names if class_names else _VOC_CLASSES)
        }
        records: list[DetectionRecord] = []
        for xml_path in sorted(source.glob("*.xml")):
            root = ET.parse(xml_path).getroot()
            filename_el = root.find("filename")
            if filename_el is None or not filename_el.text:
                raise ValueError(f"VOC file {xml_path} has no <filename>.")
            boxes: list[list[float]] = []
            labels: list[int] = []
            for obj in root.findall("object"):
                name = obj.findtext("name")
                if name not in name_to_id:
                    raise ValueError(
                        f"VOC class {name!r} in {xml_path.name} is not in the "
                        f"class list {sorted(name_to_id)}."
                    )
                box = obj.find("bndbox")
                if box is None:
                    raise ValueError(f"VOC object in {xml_path.name} has no <bndbox>.")
                boxes.append(
                    [
                        _coord(box, "xmin", xml_path),
                        _coord(box, "ymin", xml_path),
                        _coord(box, "xmax", xml_path),
                        _coord(box, "ymax", xml_path),
                    ]
                )
                labels.append(name_to_id[name])
            records.append({"sample_id": filename_el.text, "boxes": boxes, "labels": labels})
        return records

    def to_classification_records(self, source: Path) -> list[ClassificationRecord]:
        raise ValueError("VOC is a detection-only format.")
