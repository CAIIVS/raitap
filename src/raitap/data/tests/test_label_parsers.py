"""Task 3 tests: _resolve_and_parse_labels + DirectoryLabelParser e2e."""

from __future__ import annotations

import importlib
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
    importlib.import_module("raitap.data.label_parsers")
    from hydra.core.config_store import ConfigStore

    from raitap._adapters import store
    from raitap.configs.schema import AppConfig

    store.add_to_hydra_store(overwrite_ok=True)
    ConfigStore.instance().store(name="raitap_schema", node=AppConfig)


_COMPOSED_TARGET = "raitap.data.label_parsers.directory.DirectoryLabelParser"


def test_integration_compose_data_labels_directory() -> None:
    """Composing +data/labels=directory lands cfg.data.labels._target_ at the FQN."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    _register_labels_group()
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="raitap_schema", overrides=["+data/labels=directory"])
    assert cfg.data.labels._target_ == _COMPOSED_TARGET


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
