"""Task 3 tests: _resolve_and_parse_labels + DirectoryLabelParser e2e."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from raitap.configs.schema import AppConfig, DirectoryLabelsConfig
from raitap.data.data import _resolve_and_parse_labels
from raitap.types import TaskKind


def _make_cfg(
    *,
    labels: object = None,
    source: str | None = None,
    class_names: list[str] | None = None,
) -> AppConfig:
    """Build a minimal AppConfig-shaped namespace for unit tests."""
    data_ns = SimpleNamespace(labels=labels, source=source)
    model_ns = SimpleNamespace(class_names=class_names)
    return cast("AppConfig", SimpleNamespace(data=data_ns, model=model_ns))


def test_resolve_returns_none_when_labels_is_none() -> None:
    cfg = _make_cfg(labels=None)
    result = _resolve_and_parse_labels(
        cfg, task_kind=TaskKind.classification, tensor=None, sample_ids=None
    )
    assert result is None


def test_directory_parser_e2e_returns_label_tensor() -> None:
    """DirectoryLabelParser derives class index from top-level folder name."""
    import torch

    cfg = _make_cfg(labels=DirectoryLabelsConfig())
    sample_ids = ["cat/a.jpg", "dog/b.jpg"]
    result = _resolve_and_parse_labels(
        cfg, task_kind=TaskKind.classification, tensor=None, sample_ids=sample_ids
    )
    assert result is not None
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.long
    # "cat" < "dog" alphabetically -> cat=0, dog=1
    assert result.tolist() == [0, 1]


def test_directory_parser_raises_for_unsupported_task() -> None:
    cfg = _make_cfg(labels=DirectoryLabelsConfig())
    sample_ids = ["cat/a.jpg", "dog/b.jpg"]
    with pytest.raises(ValueError, match="does not support task_kind"):
        _resolve_and_parse_labels(
            cfg, task_kind=TaskKind.detection, tensor=None, sample_ids=sample_ids
        )


def test_directory_parser_returns_none_for_no_sample_ids() -> None:
    """No sample_ids -> returns None with a warning (graceful degradation)."""
    cfg = _make_cfg(labels=DirectoryLabelsConfig())
    result = _resolve_and_parse_labels(
        cfg, task_kind=TaskKind.classification, tensor=None, sample_ids=None
    )
    assert result is None


def test_directory_parser_returns_none_for_flat_layout() -> None:
    """Samples directly under root (no class subdir) -> None with warning."""
    cfg = _make_cfg(labels=DirectoryLabelsConfig())
    sample_ids = ["a.jpg", "b.jpg"]
    result = _resolve_and_parse_labels(
        cfg, task_kind=TaskKind.classification, tensor=None, sample_ids=sample_ids
    )
    assert result is None


# --- Integration: full hydra compose path ---


def _register_labels_group() -> None:
    # Canonical registration (same path as production and the rest of the suite).
    # A direct ``store.add_to_hydra_store(overwrite_ok=True)`` flush here would
    # clobber other groups' short ``_target_`` schema nodes and break later tests
    # (e.g. reporting compose asserting ``_target_ == "PDFReporter"``).
    from raitap.configs import register_configs

    register_configs()


_COMPOSED_TARGET = "raitap.data.label_parsers.directory.DirectoryLabelParser"


def test_integration_compose_data_labels_directory() -> None:
    """Composing +data/labels=directory lands cfg.data.labels.use, resolvable to the FQN."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from raitap.configs.registry_resolve import resolve_target_fqn

    _register_labels_group()
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="raitap_schema", overrides=["+data/labels=directory"])
    assert cfg.data.labels.use == "directory"
    assert resolve_target_fqn("data/labels", cfg.data.labels.use) == _COMPOSED_TARGET


# --- Task 4: TabularLabelParser ---


def _write_csv(path: object, content: str) -> None:
    import pathlib

    pathlib.Path(str(path)).write_text(content, encoding="utf-8")


def test_tabular_parser_e2e_via_resolve_and_parse_labels(tmp_path: object) -> None:
    """CSV with image,label rows + sample_ids -> aligned long tensor via resolve."""
    import pathlib

    import torch

    from raitap.configs.schema import TabularLabelsConfig
    from raitap.data.data import _resolve_and_parse_labels

    csv_path = pathlib.Path(str(tmp_path)) / "labels.csv"
    _write_csv(csv_path, "image,label\nb.jpg,1\na.jpg,0\n")

    cfg = _make_cfg(
        labels=TabularLabelsConfig(
            source=str(csv_path),
            id_column="image",
        )
    )
    sample_ids = ["a.jpg", "b.jpg"]
    result = _resolve_and_parse_labels(
        cfg, task_kind=TaskKind.classification, tensor=None, sample_ids=sample_ids
    )
    assert result is not None
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.long
    # a.jpg -> label 0, b.jpg -> label 1
    assert result.tolist() == [0, 1]


def test_tabular_parser_direct_unit(tmp_path: object) -> None:
    """Direct TabularLabelParser.parse unit test without cfg dispatch."""
    import pathlib

    import torch

    from raitap.data.label_parsers.tabular import TabularLabelParser

    csv_path = pathlib.Path(str(tmp_path)) / "labels.csv"
    _write_csv(csv_path, "image,label\na.jpg,0\nb.jpg,1\n")

    parser = TabularLabelParser(source=str(csv_path), id_column="image")
    result = parser.parse(
        task_kind=TaskKind.classification,
        tensor=None,
        sample_ids=["a.jpg", "b.jpg"],
        data_source=None,
        class_names=None,
    )
    assert result is not None
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.long
    assert result.tolist() == [0, 1]


# --- Task 5: CocoLabelParser ---


def _write_json(path: object, data: object) -> None:
    import json
    import pathlib

    pathlib.Path(str(path)).write_text(json.dumps(data), encoding="utf-8")


def _coco_detection_fixture(tmp_path: object) -> object:
    """Two-image COCO with one annotated image and one empty image."""
    import pathlib

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
    p = pathlib.Path(str(tmp_path)) / "instances.json"
    _write_json(p, coco)
    return p


def _coco_classification_fixture(tmp_path: object) -> object:
    """Two-image COCO for classification (one category per image)."""
    import pathlib

    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg"},
            {"id": 2, "file_name": "b.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 0, "bbox": [0, 0, 1, 1]},
            {"image_id": 2, "category_id": 4, "bbox": [0, 0, 1, 1]},
        ],
        "categories": [{"id": 0, "name": "x"}, {"id": 4, "name": "y"}],
    }
    p = pathlib.Path(str(tmp_path)) / "cls.json"
    _write_json(p, coco)
    return p


def test_coco_parser_detection_direct(tmp_path: object) -> None:
    """CocoLabelParser.parse detection: boxes xyxy, labels, empty-image shape."""
    import torch

    from raitap.data.label_parsers.coco import CocoLabelParser

    labels_path = _coco_detection_fixture(tmp_path)
    parser = CocoLabelParser(source=str(labels_path))
    tensor = [object(), object()]  # two samples
    result = parser.parse(
        task_kind=TaskKind.detection,
        tensor=tensor,
        sample_ids=["a.jpg", "b.jpg"],
        data_source=None,
        class_names=None,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    # a.jpg: two boxes, xyxy conversion
    expected_boxes = torch.tensor([[10.0, 20.0, 40.0, 60.0], [0.0, 0.0, 5.0, 5.0]])
    assert torch.equal(result[0]["boxes"], expected_boxes)
    assert torch.equal(result[0]["labels"], torch.tensor([3, 5]))
    # b.jpg: empty annotation -> (0, 4) boxes, (0,) labels
    assert result[1]["boxes"].shape == (0, 4)
    assert result[1]["labels"].shape == (0,)


def test_coco_parser_classification_direct(tmp_path: object) -> None:
    """CocoLabelParser.parse classification: long tensor of category ids."""
    import torch

    from raitap.data.label_parsers.coco import CocoLabelParser

    labels_path = _coco_classification_fixture(tmp_path)
    parser = CocoLabelParser(source=str(labels_path))
    result = parser.parse(
        task_kind=TaskKind.classification,
        tensor=None,
        sample_ids=["a.jpg", "b.jpg"],
        data_source=None,
        class_names=None,
    )
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.long
    assert result.tolist() == [0, 4]


def test_coco_parser_classification_rejects_multiple_categories(tmp_path: object) -> None:
    """Classification parse raises ValueError when an image has >1 categories."""
    import pathlib

    from raitap.data.label_parsers.coco import CocoLabelParser

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 1, 1]},
            {"image_id": 1, "category_id": 5, "bbox": [0, 0, 1, 1]},
        ],
        "categories": [{"id": 3, "name": "car"}, {"id": 5, "name": "dog"}],
    }
    p = pathlib.Path(str(tmp_path)) / "multi.json"
    _write_json(p, coco)
    parser = CocoLabelParser(source=str(p))
    with pytest.raises(ValueError, match="exactly one category per image"):
        parser.parse(
            task_kind=TaskKind.classification,
            tensor=None,
            sample_ids=["a.jpg"],
            data_source=None,
            class_names=None,
        )


def test_coco_parser_detection_e2e_via_resolve(tmp_path: object) -> None:
    """Detection e2e: _resolve_and_parse_labels with CocoLabelsConfig."""
    import torch

    from raitap.configs.schema import CocoLabelsConfig
    from raitap.data.data import _resolve_and_parse_labels

    labels_path = _coco_detection_fixture(tmp_path)
    cfg = _make_cfg(labels=CocoLabelsConfig(source=str(labels_path)))
    tensor = [object(), object()]
    result = _resolve_and_parse_labels(
        cfg,
        task_kind=TaskKind.detection,
        tensor=tensor,
        sample_ids=["a.jpg", "b.jpg"],
    )
    assert isinstance(result, list)
    assert len(result) == 2
    expected_boxes = torch.tensor([[10.0, 20.0, 40.0, 60.0], [0.0, 0.0, 5.0, 5.0]])
    assert torch.equal(result[0]["boxes"], expected_boxes)
    assert torch.equal(result[0]["labels"], torch.tensor([3, 5]))
    assert result[1]["boxes"].shape == (0, 4)


def test_coco_parser_classification_e2e_via_resolve(tmp_path: object) -> None:
    """Classification e2e: _resolve_and_parse_labels with CocoLabelsConfig."""
    import torch

    from raitap.configs.schema import CocoLabelsConfig
    from raitap.data.data import _resolve_and_parse_labels

    labels_path = _coco_classification_fixture(tmp_path)
    cfg = _make_cfg(labels=CocoLabelsConfig(source=str(labels_path)))
    result = _resolve_and_parse_labels(
        cfg,
        task_kind=TaskKind.classification,
        tensor=None,
        sample_ids=["a.jpg", "b.jpg"],
    )
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.long
    assert result.tolist() == [0, 4]


# --- Task 6: YoloLabelParser ---


def _make_yolo_fixture(
    tmp_path: object,
) -> tuple[object, object]:
    """Create a minimal YOLO label dir + image dir with two images.

    Returns (labels_dir, image_dir). Images are 200x100 px.
    Each .txt has one box: class 0, cx=0.5, cy=0.5, w=0.6, h=0.1.
    Denormalised: x1=(0.5-0.3)*200=40, y1=(0.5-0.05)*100=45,
                  x2=(0.5+0.3)*200=160, y2=(0.5+0.05)*100=55.
    """
    import pathlib

    from PIL import Image as PILImage

    tmp = pathlib.Path(str(tmp_path))
    labels_dir = tmp / "labels"
    labels_dir.mkdir()
    image_dir = tmp / "images"
    image_dir.mkdir()

    for stem in ("a", "b"):
        img = PILImage.new("RGB", (200, 100))
        img.save(image_dir / f"{stem}.jpg")
        (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.6 0.1\n", encoding="utf-8")

    return labels_dir, image_dir


def test_yolo_parser_unit(tmp_path: object) -> None:
    """YoloLabelParser.parse: boxes denormalised via PIL image size."""
    from raitap.data.label_parsers.yolo import YoloLabelParser

    labels_dir, image_dir = _make_yolo_fixture(tmp_path)
    parser = YoloLabelParser(source=str(labels_dir))

    tensor = [object(), object()]
    result = parser.parse(
        task_kind=TaskKind.detection,
        tensor=tensor,
        sample_ids=["a.jpg", "b.jpg"],
        data_source=str(image_dir),
        class_names=None,
    )

    assert isinstance(result, list)
    assert len(result) == 2
    # IEEE-754: (0.5+0.05)*100 = 55.00000000000001 -> use pytest.approx
    assert result[0]["boxes"][0].tolist() == pytest.approx([40.0, 45.0, 160.0, (0.5 + 0.05) * 100])
    assert result[0]["labels"].tolist() == [0]
    assert result[1]["boxes"].shape == (1, 4)


def test_yolo_parser_raises_when_data_source_none(tmp_path: object) -> None:
    """parse raises ValueError when data_source is None (no image dir)."""
    from raitap.data.label_parsers.yolo import YoloLabelParser

    labels_dir, _ = _make_yolo_fixture(tmp_path)
    parser = YoloLabelParser(source=str(labels_dir))
    with pytest.raises(ValueError, match=r"data\.source"):
        parser.parse(
            task_kind=TaskKind.detection,
            tensor=[object()],
            sample_ids=None,
            data_source=None,
            class_names=None,
        )


def test_yolo_parser_e2e_via_resolve(tmp_path: object) -> None:
    """E2E: _resolve_and_parse_labels with YoloLabelsConfig + real image dir.

    Exercises image_dir resolution through the dispatch (gap #1).
    """
    from raitap.configs.schema import YoloLabelsConfig
    from raitap.data.data import _resolve_and_parse_labels

    labels_dir, image_dir = _make_yolo_fixture(tmp_path)

    cfg = _make_cfg(
        labels=YoloLabelsConfig(source=str(labels_dir)),
        source=str(image_dir),
    )
    tensor = [object(), object()]
    result = _resolve_and_parse_labels(
        cfg,
        task_kind=TaskKind.detection,
        tensor=tensor,
        sample_ids=["a.jpg", "b.jpg"],
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["boxes"][0].tolist() == pytest.approx([40.0, 45.0, 160.0, (0.5 + 0.05) * 100])
    assert result[0]["labels"].tolist() == [0]
    assert result[1]["boxes"].shape == (1, 4)
    assert result[1]["labels"].tolist() == [0]


# --- Task 7: VocLabelParser ---


def _write_voc_xml(path: object, filename: str, objects: list[dict]) -> None:
    """Write a minimal Pascal-VOC XML file."""
    import pathlib

    lines = [
        "<annotation>",
        f"  <filename>{filename}</filename>",
    ]
    for obj in objects:
        lines += [
            "  <object>",
            f"    <name>{obj['name']}</name>",
        ]
        if obj.get("bndbox") is not None:
            b = obj["bndbox"]
            lines += [
                "    <bndbox>",
                f"      <xmin>{b[0]}</xmin>",
                f"      <ymin>{b[1]}</ymin>",
                f"      <xmax>{b[2]}</xmax>",
                f"      <ymax>{b[3]}</ymax>",
                "    </bndbox>",
            ]
        lines.append("  </object>")
    lines.append("</annotation>")
    pathlib.Path(str(path)).write_text("\n".join(lines), encoding="utf-8")


def _make_voc_fixture(tmp_path: object) -> object:
    """Two-image VOC dir with class_names=['background','person','car'].

    a.jpg: person at [10,20,30,40], car at [5,5,15,15].
    b.jpg: person at [0,0,50,50].
    """
    import pathlib

    tmp = pathlib.Path(str(tmp_path))
    voc_dir = tmp / "voc_labels"
    voc_dir.mkdir()
    _write_voc_xml(
        voc_dir / "a.xml",
        "a.jpg",
        [
            {"name": "person", "bndbox": [10, 20, 30, 40]},
            {"name": "car", "bndbox": [5, 5, 15, 15]},
        ],
    )
    _write_voc_xml(
        voc_dir / "b.xml",
        "b.jpg",
        [{"name": "person", "bndbox": [0, 0, 50, 50]}],
    )
    return voc_dir


def test_voc_parser_unit_with_class_names(tmp_path: object) -> None:
    """VocLabelParser.parse: person->1, car->2 with explicit class_names arg."""
    import torch

    from raitap.data.label_parsers.voc import VocLabelParser

    voc_dir = _make_voc_fixture(tmp_path)
    parser = VocLabelParser(source=str(voc_dir))
    class_names = ["background", "person", "car"]
    tensor = [object(), object()]
    result = parser.parse(
        task_kind=TaskKind.detection,
        tensor=tensor,
        sample_ids=["a.jpg", "b.jpg"],
        data_source=None,
        class_names=class_names,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    # a.jpg: person(1), car(2)
    expected_boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0], [5.0, 5.0, 15.0, 15.0]])
    assert torch.equal(result[0]["boxes"], expected_boxes)
    assert torch.equal(result[0]["labels"], torch.tensor([1, 2]))
    # b.jpg: person(1)
    assert torch.equal(result[1]["boxes"], torch.tensor([[0.0, 0.0, 50.0, 50.0]]))
    assert torch.equal(result[1]["labels"], torch.tensor([1]))


def test_voc_parser_raises_on_missing_bndbox(tmp_path: object) -> None:
    """parse raises ValueError when <object> has no <bndbox>."""
    import pathlib

    from raitap.data.label_parsers.voc import VocLabelParser

    tmp = pathlib.Path(str(tmp_path))
    voc_dir = tmp / "voc_no_box"
    voc_dir.mkdir()
    _write_voc_xml(
        voc_dir / "bad.xml",
        "bad.jpg",
        [{"name": "person"}],  # no bndbox key -> not written
    )
    parser = VocLabelParser(source=str(voc_dir))
    with pytest.raises(ValueError, match="no <bndbox>"):
        parser.parse(
            task_kind=TaskKind.detection,
            tensor=[object()],
            sample_ids=["bad.jpg"],
            data_source=None,
            class_names=["person"],
        )


def test_voc_parser_e2e_class_names_from_model(tmp_path: object) -> None:
    """E2E: cfg.model.class_names supplies mapping; person->1 via _resolve_and_parse_labels."""
    import torch

    from raitap.configs.schema import VocLabelsConfig
    from raitap.data.data import _resolve_and_parse_labels

    voc_dir = _make_voc_fixture(tmp_path)
    # class_names on the config is None; model supplies it instead
    cfg = _make_cfg(
        labels=VocLabelsConfig(source=str(voc_dir)),
        class_names=["background", "person", "car"],
    )
    tensor = [object(), object()]
    result = _resolve_and_parse_labels(
        cfg,
        task_kind=TaskKind.detection,
        tensor=tensor,
        sample_ids=["a.jpg", "b.jpg"],
    )
    assert isinstance(result, list)
    assert len(result) == 2
    assert torch.equal(result[0]["labels"], torch.tensor([1, 2]))
    assert torch.equal(result[1]["labels"], torch.tensor([1]))


def test_voc_parser_own_class_names_takes_precedence(tmp_path: object) -> None:
    """Parser's VocLabelsConfig.class_names overrides model's class_names."""
    import torch

    from raitap.configs.schema import VocLabelsConfig
    from raitap.data.data import _resolve_and_parse_labels

    voc_dir = _make_voc_fixture(tmp_path)
    # Parser config has class_names; model has a different (wrong) mapping
    cfg = _make_cfg(
        labels=VocLabelsConfig(
            source=str(voc_dir),
            class_names=["background", "person", "car"],
        ),
        class_names=["car", "background", "person"],  # different order -> would give wrong ids
    )
    tensor = [object(), object()]
    result = _resolve_and_parse_labels(
        cfg,
        task_kind=TaskKind.detection,
        tensor=tensor,
        sample_ids=["a.jpg", "b.jpg"],
    )
    assert isinstance(result, list)
    # Parser's own list wins: person->1, car->2
    assert torch.equal(result[0]["labels"], torch.tensor([1, 2]))


# --- Task 8: detection id_strategy parity ---


def _coco_detection_nested_fixture(tmp_path: object) -> object:
    """COCO with file_name='a.jpg' (no subdir) but discovered sample_ids=['sub/a.jpg']."""
    import pathlib

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": 2, "bbox": [1, 2, 3, 4]},
        ],
        "categories": [{"id": 2, "name": "cat"}],
    }
    p = pathlib.Path(str(tmp_path)) / "nested.json"
    _write_json(p, coco)
    return p


def test_coco_detection_nested_sample_ids_with_stem_strategy(tmp_path: object) -> None:
    """Gap #2: COCO record 'a.jpg' matches discovered 'sub/a.jpg' via id_strategy='stem'."""
    import torch

    from raitap.data.label_parsers.coco import CocoLabelParser
    from raitap.data.types import IdStrategy

    labels_path = _coco_detection_nested_fixture(tmp_path)
    parser = CocoLabelParser(source=str(labels_path), id_strategy=IdStrategy.stem)
    tensor = [object()]
    result = parser.parse(
        task_kind=TaskKind.detection,
        tensor=tensor,
        sample_ids=["sub/a.jpg"],
        data_source=None,
        class_names=None,
    )
    assert isinstance(result, list)
    assert len(result) == 1
    # bbox [1,2,3,4] -> xyxy [1, 2, 1+3, 2+4] = [1, 2, 4, 6]
    expected_boxes = torch.tensor([[1.0, 2.0, 4.0, 6.0]])
    assert torch.equal(result[0]["boxes"], expected_boxes)
    assert torch.equal(result[0]["labels"], torch.tensor([2]))


def test_coco_detection_exact_match_regression(tmp_path: object) -> None:
    """Regression: exact-match ids still align under id_strategy='auto'."""
    import torch

    from raitap.data.label_parsers.coco import CocoLabelParser
    from raitap.data.types import IdStrategy

    labels_path = _coco_detection_fixture(tmp_path)
    parser = CocoLabelParser(source=str(labels_path), id_strategy=IdStrategy.auto)
    tensor = [object(), object()]
    result = parser.parse(
        task_kind=TaskKind.detection,
        tensor=tensor,
        sample_ids=["a.jpg", "b.jpg"],
        data_source=None,
        class_names=None,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    expected_boxes = torch.tensor([[10.0, 20.0, 40.0, 60.0], [0.0, 0.0, 5.0, 5.0]])
    assert torch.equal(result[0]["boxes"], expected_boxes)
    assert torch.equal(result[0]["labels"], torch.tensor([3, 5]))
    assert result[1]["boxes"].shape == (0, 4)
    assert result[1]["labels"].shape == (0,)


def test_coco_parser_classification_returns_none_without_sample_ids(tmp_path: object) -> None:
    """COCO classification with no sample_ids -> None (predictions-as-targets),
    not a silent empty label tensor."""
    from raitap.data.label_parsers.coco import CocoLabelParser

    p = _coco_classification_fixture(tmp_path)
    parser = CocoLabelParser(source=str(p))
    result = parser.parse(
        task_kind=TaskKind.classification,
        tensor=None,
        sample_ids=None,
        data_source=None,
        class_names=None,
    )
    assert result is None


def test_yolo_parser_raises_on_malformed_line(tmp_path: object) -> None:
    """A YOLO label line with fewer than 5 fields raises a clear ValueError
    naming the file, not a bare unpack error."""
    import pathlib

    from raitap.data.label_parsers.yolo import YoloLabelParser

    labels_dir, image_dir = _make_yolo_fixture(tmp_path)
    (pathlib.Path(str(labels_dir)) / "a.txt").write_text("0 0.5 0.5\n", encoding="utf-8")

    parser = YoloLabelParser(source=str(labels_dir))
    with pytest.raises(ValueError, match=r"expected 5"):
        parser.parse(
            task_kind=TaskKind.detection,
            tensor=[object(), object()],
            sample_ids=["a.jpg", "b.jpg"],
            data_source=str(image_dir),
            class_names=None,
        )


def test_yolo_parser_nested_split_layout(tmp_path: object) -> None:
    """Recursive YOLO split layout (labels/train/x.txt, images/train/x.jpg).

    sample_id preserves the image-relative path so id_strategy=relative_path
    matches nested discovered sample ids.
    """
    import pathlib

    from PIL import Image as PILImage

    from raitap.data.label_parsers.yolo import YoloLabelParser
    from raitap.data.types import IdStrategy

    tmp = pathlib.Path(str(tmp_path))
    labels_dir = tmp / "labels" / "train"
    labels_dir.mkdir(parents=True)
    image_dir = tmp / "images"
    (image_dir / "train").mkdir(parents=True)
    PILImage.new("RGB", (200, 100)).save(image_dir / "train" / "a.jpg")
    (labels_dir / "a.txt").write_text("0 0.5 0.5 0.6 0.1\n", encoding="utf-8")

    parser = YoloLabelParser(source=str(tmp / "labels"), id_strategy=IdStrategy.relative_path)
    result = parser.parse(
        task_kind=TaskKind.detection,
        tensor=[object()],
        sample_ids=["train/a.jpg"],
        data_source=str(image_dir),
        class_names=None,
    )
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["labels"].tolist() == [0]
    # box denormalised against the 200x100 image
    assert result[0]["boxes"][0].tolist() == pytest.approx([40.0, 45.0, 160.0, 55.0])


def test_yolo_parser_raises_on_non_integer_class(tmp_path: object) -> None:
    """A non-integer YOLO class index fails loudly instead of truncating."""
    import pathlib

    from raitap.data.label_parsers.yolo import YoloLabelParser

    labels_dir, image_dir = _make_yolo_fixture(tmp_path)
    (pathlib.Path(str(labels_dir)) / "a.txt").write_text("1.9 0.5 0.5 0.6 0.1\n", encoding="utf-8")

    parser = YoloLabelParser(source=str(labels_dir))
    with pytest.raises(ValueError, match="class index must be an integer"):
        parser.parse(
            task_kind=TaskKind.detection,
            tensor=[object(), object()],
            sample_ids=["a.jpg", "b.jpg"],
            data_source=str(image_dir),
            class_names=None,
        )


def test_coco_classification_misaligned_returns_none(tmp_path: object) -> None:
    """COCO classification with unmatchable sample ids warns and returns None
    (predictions-as-targets fallback), matching the tabular parser."""
    from raitap.data.label_parsers.coco import CocoLabelParser

    p = _coco_classification_fixture(tmp_path)
    parser = CocoLabelParser(source=str(p))
    result = parser.parse(
        task_kind=TaskKind.classification,
        tensor=None,
        sample_ids=["not-in-labels.jpg"],
        data_source=None,
        class_names=None,
    )
    assert result is None
