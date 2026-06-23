import pytest
from raitap.data.types import LabelFormat
from raitap.configs.schema import LabelsConfig
from raitap.data.label_formats import (
    LABEL_FORMAT_ADAPTERS,
    label_format,
    resolve_label_format_adapter,
)
from raitap.types import TaskKind


def test_label_format_members_are_string_values():
    assert LabelFormat.native == "native"
    assert {f.value for f in LabelFormat} == {"native", "coco", "yolo", "voc"}


def test_labels_config_defaults_to_native_format():
    assert LabelsConfig().format is LabelFormat.native


def test_label_format_decorator_registers_instance():
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


def test_registry_rejects_unknown_native():
    with pytest.raises(ValueError, match="No adapter"):
        resolve_label_format_adapter(LabelFormat.native, task_kind=TaskKind.detection)


def test_registry_resolves_supported_task():
    adapter = resolve_label_format_adapter(LabelFormat.coco, task_kind=TaskKind.detection)
    assert adapter.format is LabelFormat.coco
    assert TaskKind.detection in adapter.supported_tasks


def test_registry_rejects_unsupported_task():
    with pytest.raises(ValueError, match="does not support task"):
        resolve_label_format_adapter(LabelFormat.yolo, task_kind=TaskKind.classification)


def test_coco_detection_records(tmp_path):
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


def test_coco_classification_records(tmp_path):
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


def test_coco_classification_rejects_zero_categories(tmp_path):
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


def test_coco_classification_rejects_multiple_categories(tmp_path):
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


def test_yolo_detection_records(tmp_path):
    from PIL import Image
    from raitap.data.adapters.yolo import YoloAdapter

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (100, 200)).save(image_dir / "a.jpg")  # w=100, h=200

    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    # class=2, cx=0.5 cy=0.5 w=0.2 h=0.1  -> center (50,100), box 20x20px
    (label_dir / "a.txt").write_text("2 0.5 0.5 0.2 0.1\n")

    records = YoloAdapter().to_detection_records(
        label_dir, image_dir=image_dir, class_names=None
    )
    assert len(records) == 1
    rec = records[0]
    assert rec["sample_id"] == "a.jpg"
    assert rec["labels"] == [2]
    # x1 = (0.5-0.1)*100=40, y1=(0.5-0.05)*200=90, x2=60, y2=110
    assert len(rec["boxes"]) == 1
    assert rec["boxes"][0] == pytest.approx([40.0, 90.0, 60.0, 110.0])
