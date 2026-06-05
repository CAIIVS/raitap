"""Tests for DetectionFamily.load_labels — list[dict] per-sample boxes + labels."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from raitap.data.data import Data
from raitap.task_families.detection import DetectionFamily

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig


def _write_detection_labels_json(path: Path) -> None:
    payload = [
        {
            "sample_id": "img_0",
            "boxes": [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 12.0, 12.0]],
            "labels": [1, 2],
        },
        {
            "sample_id": "img_1",
            "boxes": [],
            "labels": [],
        },
        {
            "sample_id": "img_2",
            "boxes": [[3.0, 3.0, 6.0, 6.0]],
            "labels": [1],
        },
    ]
    path.write_text(json.dumps(payload))


def _stub_cfg(labels_source: str | None = None, labels_kind: str | None = None) -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            data=SimpleNamespace(
                labels=SimpleNamespace(source=labels_source, kind=labels_kind),
            ),
        ),
    )


def _make_data(*, num_samples: int = 3, sample_ids: list[str] | None = None) -> Data:
    data = Data.__new__(Data)
    data.tensor = torch.zeros(num_samples, 3, 4, 4)
    data.sample_ids = sample_ids
    return data


def test_load_detection_labels_returns_list_of_dicts(tmp_path: Path) -> None:
    labels_path = tmp_path / "boxes.json"
    _write_detection_labels_json(labels_path)
    cfg = _stub_cfg(labels_source=str(labels_path), labels_kind="detection")

    data = _make_data(num_samples=3)
    out = DetectionFamily().load_labels(cfg, tensor=data.tensor, sample_ids=data.sample_ids)
    assert out is not None
    assert isinstance(out, list)
    assert len(out) == 3
    assert torch.equal(
        out[0]["boxes"],
        torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 12.0, 12.0]]),
    )
    assert torch.equal(out[0]["labels"], torch.tensor([1, 2], dtype=torch.int64))
    assert out[1]["boxes"].shape == (0, 4)
    assert out[1]["labels"].shape == (0,)
    assert out[1]["boxes"].dtype == torch.float32
    assert out[1]["labels"].dtype == torch.int64


def test_load_detection_labels_aligns_by_sample_id_when_present(tmp_path: Path) -> None:
    """Reordered labels file is rewritten to match self.sample_ids ordering."""
    labels_path = tmp_path / "boxes.json"
    # Write records out of order vs sample_ids.
    payload = [
        {"sample_id": "img_2", "boxes": [[3.0, 3.0, 6.0, 6.0]], "labels": [9]},
        {"sample_id": "img_0", "boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": [7]},
        {"sample_id": "img_1", "boxes": [], "labels": []},
    ]
    labels_path.write_text(json.dumps(payload))
    cfg = _stub_cfg(labels_source=str(labels_path), labels_kind="detection")

    data = _make_data(num_samples=3, sample_ids=["img_0", "img_1", "img_2"])
    out = DetectionFamily().load_labels(cfg, tensor=data.tensor, sample_ids=data.sample_ids)
    assert out is not None
    assert int(out[0]["labels"].item()) == 7
    assert out[1]["labels"].numel() == 0
    assert int(out[2]["labels"].item()) == 9


def test_load_detection_labels_rejects_missing_sample_id_entries(tmp_path: Path) -> None:
    labels_path = tmp_path / "boxes.json"
    payload = [
        {"sample_id": "img_0", "boxes": [], "labels": []},
        # img_1 missing
        {"sample_id": "img_2", "boxes": [], "labels": []},
    ]
    labels_path.write_text(json.dumps(payload))
    cfg = _stub_cfg(labels_source=str(labels_path), labels_kind="detection")

    data = _make_data(num_samples=3, sample_ids=["img_0", "img_1", "img_2"])
    with pytest.raises(ValueError, match="missing entries"):
        DetectionFamily().load_labels(cfg, tensor=data.tensor, sample_ids=data.sample_ids)


def test_load_detection_labels_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    labels_path = tmp_path / "boxes.json"
    payload = [
        {"sample_id": "img_0", "boxes": [], "labels": []},
        {"sample_id": "img_0", "boxes": [], "labels": []},
        {"sample_id": "img_1", "boxes": [], "labels": []},
    ]
    labels_path.write_text(json.dumps(payload))
    cfg = _stub_cfg(labels_source=str(labels_path), labels_kind="detection")

    data = _make_data(num_samples=2, sample_ids=["img_0", "img_1"])
    with pytest.raises(ValueError, match="duplicate sample_id"):
        DetectionFamily().load_labels(cfg, tensor=data.tensor, sample_ids=data.sample_ids)


def test_load_detection_labels_rejects_record_missing_sample_id_field(tmp_path: Path) -> None:
    labels_path = tmp_path / "boxes.json"
    payload = [
        {"boxes": [], "labels": []},  # no sample_id
    ]
    labels_path.write_text(json.dumps(payload))
    cfg = _stub_cfg(labels_source=str(labels_path), labels_kind="detection")

    data = _make_data(num_samples=1, sample_ids=["img_0"])
    with pytest.raises(ValueError, match="missing 'sample_id'"):
        DetectionFamily().load_labels(cfg, tensor=data.tensor, sample_ids=data.sample_ids)


def test_load_detection_labels_rejects_wrong_length_when_no_sample_ids(tmp_path: Path) -> None:
    """Without sample_ids, record count must match dataset length exactly."""
    labels_path = tmp_path / "boxes.json"
    _write_detection_labels_json(labels_path)  # 3 records
    cfg = _stub_cfg(labels_source=str(labels_path), labels_kind="detection")

    data = _make_data(num_samples=5)  # dataset bigger than labels
    with pytest.raises(ValueError, match="5 samples"):
        DetectionFamily().load_labels(cfg, tensor=data.tensor, sample_ids=data.sample_ids)


def test_load_detection_labels_rejects_mismatched_box_label_counts(tmp_path: Path) -> None:
    labels_path = tmp_path / "bad.json"
    labels_path.write_text(
        json.dumps([{"sample_id": "x", "boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": [1, 2]}])
    )
    cfg = _stub_cfg(labels_source=str(labels_path), labels_kind="detection")

    data = _make_data(num_samples=1)
    with pytest.raises(ValueError, match="boxes and labels"):
        DetectionFamily().load_labels(cfg, tensor=data.tensor, sample_ids=data.sample_ids)


def test_load_detection_labels_rejects_non_list_root(tmp_path: Path) -> None:
    labels_path = tmp_path / "bad.json"
    labels_path.write_text(json.dumps({"not": "a list"}))
    cfg = _stub_cfg(labels_source=str(labels_path), labels_kind="detection")

    data = _make_data(num_samples=1)
    with pytest.raises(ValueError, match="must be a JSON array"):
        DetectionFamily().load_labels(cfg, tensor=data.tensor, sample_ids=data.sample_ids)


def test_load_detection_labels_returns_none_when_no_source_configured(tmp_path: Path) -> None:
    cfg = _stub_cfg(labels_source=None, labels_kind="detection")
    data = _make_data(num_samples=1)
    out = DetectionFamily().load_labels(cfg, tensor=data.tensor, sample_ids=data.sample_ids)
    assert out is None


def test_load_detection_labels_raises_when_source_unresolvable(tmp_path: Path) -> None:
    cfg = _stub_cfg(labels_source=str(tmp_path / "missing.json"), labels_kind="detection")
    data = _make_data(num_samples=1)
    with pytest.raises(ValueError, match="could not be resolved"):
        DetectionFamily().load_labels(cfg, tensor=data.tensor, sample_ids=data.sample_ids)


def test_detection_loader_ignores_labels_kind_field(tmp_path: Path) -> None:
    """The family is chosen by ``task_kind``; ``DetectionFamily.load_labels``
    loads detection targets regardless of the (now-unread) ``data.labels.kind``
    value — enum member or raw string."""
    from raitap.data.types import LabelKind

    labels_path = tmp_path / "boxes.json"
    _write_detection_labels_json(labels_path)

    for kind_value in (LabelKind.detection, "detection", None):
        cfg = _stub_cfg(labels_source=str(labels_path), labels_kind=kind_value)
        data = _make_data(num_samples=3)
        out = DetectionFamily().load_labels(cfg, tensor=data.tensor, sample_ids=data.sample_ids)
        assert out is not None
        assert len(out) == 3
