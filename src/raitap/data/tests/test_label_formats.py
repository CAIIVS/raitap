from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from raitap.configs.schema import LabelsConfig
from raitap.data.label_formats import (
    LABEL_FORMAT_ADAPTERS,
    label_format,
    resolve_label_format_adapter,
)
from raitap.data.types import LabelFormat
from raitap.types import TaskKind

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig


def test_label_format_members_are_string_values() -> None:
    assert LabelFormat.native == "native"
    assert {f.value for f in LabelFormat} == {"native", "coco", "yolo", "voc"}


def test_labels_config_defaults_to_native_format() -> None:
    assert LabelsConfig().format is LabelFormat.native


def test_label_format_decorator_registers_instance() -> None:
    @label_format
    class _Dummy:
        format = LabelFormat.coco  # reuse an enum member; popped below
        supported_tasks = frozenset({TaskKind.detection})

    try:
        assert LABEL_FORMAT_ADAPTERS[LabelFormat.coco].supported_tasks == frozenset(
            {TaskKind.detection}
        )
    finally:
        LABEL_FORMAT_ADAPTERS.pop(LabelFormat.coco, None)


def test_registry_rejects_unknown_native() -> None:
    with pytest.raises(ValueError, match="No adapter"):
        resolve_label_format_adapter(LabelFormat.native, task_kind=TaskKind.detection)


def test_registry_resolves_supported_task() -> None:
    adapter = resolve_label_format_adapter(LabelFormat.coco, task_kind=TaskKind.detection)
    assert adapter.format is LabelFormat.coco
    assert TaskKind.detection in adapter.supported_tasks


def test_registry_rejects_unsupported_task() -> None:
    with pytest.raises(ValueError, match="does not support task"):
        resolve_label_format_adapter(LabelFormat.yolo, task_kind=TaskKind.classification)


def test_coco_detection_records(tmp_path: Path) -> None:
    import json

    from raitap.data.adapters.coco import CocoAdapter

    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg"},
            {"id": 2, "file_name": "b.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 3, "bbox": [10, 20, 30, 40]},
            {"image_id": 1, "category_id": 5, "bbox": [0, 0, 5, 5]},
        ],
        "categories": [{"id": 3, "name": "car"}, {"id": 5, "name": "dog"}],
    }
    p = tmp_path / "instances.json"
    p.write_text(json.dumps(coco))

    records = CocoAdapter().to_detection_records(p, image_dir=None, class_names=None)
    by_id = {r["sample_id"]: r for r in records}
    assert by_id["a.jpg"]["boxes"] == [[10, 20, 40, 60], [0, 0, 5, 5]]
    assert by_id["a.jpg"]["labels"] == [3, 5]
    assert by_id["b.jpg"] == {"sample_id": "b.jpg", "boxes": [], "labels": []}


def test_coco_classification_records(tmp_path: Path) -> None:
    import json

    from raitap.data.adapters.coco import CocoAdapter

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [{"image_id": 1, "category_id": 7, "bbox": [0, 0, 1, 1]}],
        "categories": [{"id": 7, "name": "cat"}],
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(coco))
    records = CocoAdapter().to_classification_records(p)
    assert records == [{"sample_id": "a.jpg", "label": 7}]


def test_coco_classification_rejects_zero_categories(tmp_path: Path) -> None:
    import json

    from raitap.data.adapters.coco import CocoAdapter

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [],
        "categories": [{"id": 7, "name": "cat"}],
    }
    p = tmp_path / "zero.json"
    p.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="exactly one category per image"):
        CocoAdapter().to_classification_records(p)


def test_coco_classification_rejects_multiple_categories(tmp_path: Path) -> None:
    import json

    from raitap.data.adapters.coco import CocoAdapter

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 1, 1]},
            {"image_id": 1, "category_id": 5, "bbox": [0, 0, 1, 1]},
        ],
        "categories": [{"id": 3, "name": "car"}, {"id": 5, "name": "dog"}],
    }
    p = tmp_path / "multi.json"
    p.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="exactly one category per image"):
        CocoAdapter().to_classification_records(p)


def test_yolo_detection_records(tmp_path: Path) -> None:
    from PIL import Image

    from raitap.data.adapters.yolo import YoloAdapter

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (100, 200)).save(image_dir / "a.jpg")  # w=100, h=200

    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    # class=2, cx=0.5 cy=0.5 w=0.2 h=0.1  -> center (50,100), box 20x20px
    (label_dir / "a.txt").write_text("2 0.5 0.5 0.2 0.1\n")

    records = YoloAdapter().to_detection_records(label_dir, image_dir=image_dir, class_names=None)
    assert len(records) == 1
    rec = records[0]
    assert rec["sample_id"] == "a.jpg"
    assert rec["labels"] == [2]
    # x1 = (0.5-0.1)*100=40, y1=(0.5-0.05)*200=90, x2=60, y2=110
    assert len(rec["boxes"]) == 1
    assert rec["boxes"][0] == pytest.approx([40.0, 90.0, 60.0, 110.0])


def test_voc_detection_records(tmp_path: Path) -> None:
    from raitap.data.adapters.voc import VocAdapter

    xml = """<annotation>
      <filename>a.jpg</filename>
      <object><name>person</name>
        <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>
      </object>
    </annotation>"""
    d = tmp_path / "ann"
    d.mkdir()
    (d / "a.xml").write_text(xml)

    records = VocAdapter().to_detection_records(
        d, image_dir=None, class_names=["background", "person", "car"]
    )
    assert records == [{"sample_id": "a.jpg", "boxes": [[10.0, 20.0, 30.0, 40.0]], "labels": [1]}]


def test_voc_detection_rejects_object_without_bndbox(tmp_path: Path) -> None:
    from raitap.data.adapters.voc import VocAdapter

    xml = """<annotation>
      <filename>a.jpg</filename>
      <object><name>person</name></object>
    </annotation>"""
    d = tmp_path / "ann"
    d.mkdir()
    (d / "a.xml").write_text(xml)

    with pytest.raises(ValueError, match="has no <bndbox>"):
        VocAdapter().to_detection_records(
            d, image_dir=None, class_names=["background", "person", "car"]
        )


def test_detection_load_labels_via_coco(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from types import SimpleNamespace

    import torch

    import raitap.data.data as data_mod
    from raitap.data.types import LabelFormat
    from raitap.task_families.detection import DetectionFamily

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "annotations": [{"image_id": 1, "category_id": 3, "bbox": [10, 20, 30, 40]}],
        "categories": [{"id": 3, "name": "car"}],
    }
    labels_file = tmp_path / "instances.json"
    labels_file.write_text(json.dumps(coco))

    monkeypatch.setattr(data_mod, "get_source_path", lambda source, *, kind: tmp_path / source)
    # tmp_path/"instances.json" is LABELS; tmp_path/"imgs" is DATA (unused by coco).
    cfg = cast(
        "AppConfig",
        SimpleNamespace(
            data=SimpleNamespace(
                source="imgs",
                labels=SimpleNamespace(source="instances.json", format=LabelFormat.coco),
            )
        ),
    )
    tensor = [object(), object()]  # len == 2 samples
    out = DetectionFamily().load_labels(cfg, tensor=tensor, sample_ids=["a.jpg", "b.jpg"])
    assert torch.equal(out[0]["boxes"], torch.tensor([[10.0, 20.0, 40.0, 60.0]]))
    assert torch.equal(out[0]["labels"], torch.tensor([3]))
    assert out[1]["boxes"].shape == (0, 4)


def test_classification_load_labels_via_coco(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from types import SimpleNamespace

    import torch

    import raitap.data.data as data_mod
    from raitap.data.types import LabelFormat

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": 0, "bbox": [0, 0, 1, 1]},
            {"image_id": 2, "category_id": 4, "bbox": [0, 0, 1, 1]},
        ],
        "categories": [{"id": 0, "name": "x"}, {"id": 4, "name": "y"}],
    }
    labels_file = tmp_path / "c.json"
    labels_file.write_text(json.dumps(coco))
    monkeypatch.setattr(data_mod, "get_source_path", lambda source, *, kind: tmp_path / source)
    cfg = cast(
        "AppConfig",
        SimpleNamespace(
            data=SimpleNamespace(
                source="imgs",
                labels=SimpleNamespace(
                    source="c.json", format=LabelFormat.coco, id_strategy="stem"
                ),
            )
        ),
    )
    out = data_mod.load_classification_labels(
        cfg, tensor=torch.zeros(2), sample_ids=["a.jpg", "b.jpg"]
    )
    assert out is not None
    assert torch.equal(out, torch.tensor([0, 4]))
