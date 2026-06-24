"""Pascal-VOC label parser (detection-only)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

from raitap.configs.schema import VocLabelsConfig
from raitap.data.data import SourceKind, get_source_path
from raitap.data.label_parsers.registration import label_parser
from raitap.data.types import IdStrategy
from raitap.task_families.detection import _align_detection_records
from raitap.types import TaskKind

if TYPE_CHECKING:
    from pathlib import Path

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


@label_parser(registry_name="voc", schema=VocLabelsConfig)
class VocLabelParser:
    """Parse Pascal-VOC per-image ``.xml`` for detection.

    Boxes are already ``[xmin, ymin, xmax, ymax]`` pixels. Class names map to
    ids by their position in the active name list (parser's own ``class_names``,
    else the ``class_names`` arg from ``cfg.model.class_names``, else the
    standard 20-class VOC order).
    """

    supported_tasks: frozenset[TaskKind] = frozenset({TaskKind.detection})

    def __init__(
        self,
        *,
        source: str,
        id_strategy: IdStrategy = IdStrategy.auto,
        class_names: list[str] | None = None,
    ) -> None:
        self.source = source
        self.id_strategy = id_strategy
        self.class_names = class_names

    def _to_detection_records(
        self, labels_dir: Path, name_to_id: dict[str, int]
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for xml_path in sorted(labels_dir.glob("*.xml")):
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

    def parse(
        self,
        *,
        task_kind: TaskKind,
        tensor: Any,
        sample_ids: list[str] | None,
        data_source: str | None,
        class_names: list[str] | None,
    ) -> Any:
        """Load VOC xml labels and align to sample_ids for detection."""
        labels_dir = get_source_path(self.source, kind=SourceKind.LABELS)
        # Precedence: parser's own class_names > model's class_names > _VOC_CLASSES
        active_names: list[str] | tuple[str, ...] = (
            self.class_names
            if self.class_names is not None
            else (class_names if class_names is not None else _VOC_CLASSES)
        )
        name_to_id = {name: idx for idx, name in enumerate(active_names)}
        records = self._to_detection_records(labels_dir, name_to_id)
        return _align_detection_records(
            records,
            expected=len(tensor),
            sample_ids=sample_ids,
        )
