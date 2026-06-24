"""TDD tests for the ragged-list detection data path (issue #197).

Covers:
- Detection task_kind + differently-sized images → list[Tensor] (no ValueError).
- Classification task_kind on uniform images → stacked dense tensor (regression guard).
- DetectionFamily.load_labels count validation works when the tensor is a list.
- describe() returns sane metadata for a list-valued tensor.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from PIL import Image

from raitap.configs.schema import DetectionJsonLabelsConfig
from raitap.data.data import Data, _resolve_and_parse_labels
from raitap.task_families.classification import ClassificationFamily
from raitap.task_families.detection import DetectionFamily
from raitap.types import TaskKind

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_image(path: Path, width: int, height: int) -> None:
    """Write a minimal solid-colour RGB image with the given size."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _make_config(source: str, name: str = "test_det") -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            data=SimpleNamespace(
                source=source,
                name=name,
                labels=None,
            )
        ),
    )


# ---------------------------------------------------------------------------
# Core bug regression: ragged image directory must not raise
# ---------------------------------------------------------------------------


class TestDetectionRaggedLoader:
    def test_different_sized_images_detection_returns_list(self, tmp_path: Path) -> None:
        """Detection task_kind: differently-sized images load as list[Tensor], not stacked."""
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        # Three images with different spatial sizes — np.stack would crash here.
        _write_image(data_dir / "a.jpg", width=32, height=24)
        _write_image(data_dir / "b.jpg", width=64, height=48)
        _write_image(data_dir / "c.jpg", width=16, height=16)

        cfg = _make_config(str(data_dir))
        data = Data(cfg, task_kind=TaskKind.detection)

        # Must be a list, not a stacked tensor.
        assert isinstance(data.tensor, list)
        assert len(data.tensor) == 3

        # Each element is a (C, H, W) float32 tensor in [0, 1].
        shapes = [tuple(t.shape) for t in data.tensor]
        assert (3, 24, 32) in shapes
        assert (3, 48, 64) in shapes
        assert (3, 16, 16) in shapes
        for t in data.tensor:
            assert t.dtype == torch.float32
            assert t.min() >= 0.0
            assert t.max() <= 1.0

    def test_different_sized_images_detection_preserves_native_size(self, tmp_path: Path) -> None:
        """No resize is applied — each tensor keeps its native resolution."""
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "wide.jpg", width=100, height=50)
        _write_image(data_dir / "tall.jpg", width=20, height=80)

        cfg = _make_config(str(data_dir))
        data = Data(cfg, task_kind=TaskKind.detection)

        assert isinstance(data.tensor, list)
        shapes = sorted(tuple(t.shape) for t in data.tensor)
        assert (3, 50, 100) in shapes
        assert (3, 80, 20) in shapes

    def test_single_image_file_detection_returns_list(self, tmp_path: Path) -> None:
        """A single image file also produces a list of length 1 under detection."""
        img = tmp_path / "img.jpg"
        _write_image(img, width=32, height=32)
        cfg = _make_config(str(img))
        data = Data(cfg, task_kind=TaskKind.detection)

        assert isinstance(data.tensor, list)
        assert len(data.tensor) == 1
        assert data.tensor[0].shape == (3, 32, 32)


# ---------------------------------------------------------------------------
# Classification regression guard: stacked tensor path unchanged
# ---------------------------------------------------------------------------


class TestClassificationStackedGuard:
    def test_classification_uniform_images_still_stacked(self, tmp_path: Path) -> None:
        """Default (classification) task_kind still produces a stacked dense NCHW tensor."""
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "a.jpg", width=32, height=32)
        _write_image(data_dir / "b.jpg", width=32, height=32)

        cfg = _make_config(str(data_dir))
        data = Data(cfg)  # default task_kind = classification

        assert isinstance(data.tensor, torch.Tensor)
        assert isinstance(data.tensor, torch.Tensor)
        assert data.tensor.shape == (2, 3, 32, 32)
        assert data.tensor.dtype == torch.float32

    def test_explicit_classification_task_kind_also_stacks(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "x.jpg", width=8, height=8)

        cfg = _make_config(str(data_dir))
        data = Data(cfg, task_kind=TaskKind.classification)

        assert isinstance(data.tensor, torch.Tensor)
        assert isinstance(data.tensor, torch.Tensor)
        assert data.tensor.shape == (1, 3, 8, 8)


# ---------------------------------------------------------------------------
# Detection label count validation with list tensor
# ---------------------------------------------------------------------------


class TestDetectionLabelsWithListTensor:
    def _write_labels_json(self, path: Path, n: int) -> None:
        payload = [{"sample_id": f"img_{i}", "boxes": [], "labels": []} for i in range(n)]
        path.write_text(json.dumps(payload))

    def test_detection_labels_count_matches_list_tensor(self, tmp_path: Path) -> None:
        """DetectionJsonLabelParser: len(tensor) works when tensor is a list."""
        labels_path = tmp_path / "boxes.json"
        self._write_labels_json(labels_path, n=3)

        data = Data.__new__(Data)
        # Simulate a ragged list (different sizes so a stacked tensor couldn't exist).
        data.tensor = [
            torch.zeros(3, 24, 32),
            torch.zeros(3, 48, 64),
            torch.zeros(3, 16, 16),
        ]
        data.sample_ids = ["img_0", "img_1", "img_2"]

        cfg = cast(
            "AppConfig",
            SimpleNamespace(
                data=SimpleNamespace(
                    labels=DetectionJsonLabelsConfig(source=str(labels_path)),
                    source=None,
                ),
                model=SimpleNamespace(class_names=None),
            ),
        )
        out = _resolve_and_parse_labels(
            cfg, task_kind=TaskKind.detection, tensor=data.tensor, sample_ids=data.sample_ids
        )
        assert out is not None
        assert len(out) == 3

    def test_detection_labels_count_mismatch_raises_with_list_tensor(self, tmp_path: Path) -> None:
        """Wrong record count still raises even when tensor is a list."""
        labels_path = tmp_path / "boxes.json"
        self._write_labels_json(labels_path, n=2)  # only 2 records

        data = Data.__new__(Data)
        data.tensor = [torch.zeros(3, 24, 32), torch.zeros(3, 48, 64), torch.zeros(3, 16, 16)]
        data.sample_ids = None  # force row-order path

        cfg = cast(
            "AppConfig",
            SimpleNamespace(
                data=SimpleNamespace(
                    labels=DetectionJsonLabelsConfig(source=str(labels_path)),
                    source=None,
                ),
                model=SimpleNamespace(class_names=None),
            ),
        )
        with pytest.raises(ValueError, match="3 samples"):
            _resolve_and_parse_labels(
                cfg, task_kind=TaskKind.detection, tensor=data.tensor, sample_ids=data.sample_ids
            )


# ---------------------------------------------------------------------------
# describe() with a ragged list tensor
# ---------------------------------------------------------------------------


class TestDescribeWithListTensor:
    def test_describe_ragged_list(self, tmp_path: Path) -> None:
        """describe() must not crash and must be JSON-serialisable for a list tensor."""
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "a.jpg", width=32, height=24)
        _write_image(data_dir / "b.jpg", width=64, height=48)

        cfg = _make_config(str(data_dir), name="det_data")
        data = Data(cfg, task_kind=TaskKind.detection)

        info = data.describe()

        # Must be JSON-serialisable (no tensors, no torch.Size, no np.int64, ...).
        import json as _json

        serialised = _json.dumps(info)
        reloaded = _json.loads(serialised)

        assert reloaded["num_samples"] == 2
        assert reloaded["has_labels"] is False
        assert reloaded["name"] == "det_data"
        # Shape must be flagged as ragged (string or absent).
        assert reloaded.get("shape") == "ragged" or "shape" not in reloaded
        # sample_shape must be absent for ragged data.
        assert "sample_shape" not in reloaded

    def test_describe_dense_tensor_unchanged(self, tmp_path: Path) -> None:
        """Dense-tensor path in describe() keeps the existing list[int] shape contract."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6")
        cfg = _make_config(str(csv_file))
        data = Data(cfg)

        info = data.describe()
        assert info["shape"] == [2, 3]
        assert info["num_samples"] == 2
        assert "sample_shape" in info


# ---------------------------------------------------------------------------
# Loaded-tensor contract validation
# ---------------------------------------------------------------------------


class TestValidateLoadedTensor:
    def test_detection_accepts_list_of_chw_tensors(self) -> None:
        DetectionFamily().validate_inputs([torch.zeros(3, 24, 32)])  # no raise

    def test_detection_rejects_dense_tensor(self) -> None:
        with pytest.raises(TypeError, match="must be a list"):
            DetectionFamily().validate_inputs(torch.zeros(2, 3, 8, 8))

    def test_detection_rejects_empty_list(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            DetectionFamily().validate_inputs([])

    def test_detection_rejects_non_chw_entry(self) -> None:
        with pytest.raises(ValueError, match=r"\(C, H, W\)"):
            DetectionFamily().validate_inputs([torch.zeros(24, 32)])

    def test_classification_accepts_batched_tensor(self) -> None:
        ClassificationFamily().validate_inputs(torch.zeros(4, 3, 8, 8))  # no raise

    def test_classification_rejects_list(self) -> None:
        with pytest.raises(TypeError, match="dense"):
            ClassificationFamily().validate_inputs([torch.zeros(3, 8, 8)])

    def test_classification_rejects_unbatched_tensor(self) -> None:
        with pytest.raises(ValueError, match="ndim >= 2"):
            ClassificationFamily().validate_inputs(torch.zeros(5))

    def test_classification_rejects_empty_batch(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ClassificationFamily().validate_inputs(torch.zeros(0, 3, 8, 8))
