"""Unit tests for the Data class."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

from raitap.data import Data


def _write_image(path: Path) -> None:
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _make_config(
    source: str,
    name: str = "test_data",
    labels_source: str | None = None,
    labels_id_column: str | None = None,
    labels_column: str | None = None,
    labels_encoding: str | None = None,
) -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            data=SimpleNamespace(
                source=source,
                name=name,
                labels=SimpleNamespace(
                    source=labels_source,
                    id_column=labels_id_column,
                    column=labels_column,
                    encoding=labels_encoding,
                ),
            )
        ),
    )


class TestDataConstructor:
    def test_data_loads_tensor_from_source(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1.0,2.0\n3.0,4.0")
        config = _make_config(str(csv_file))

        data = Data(config)

        assert isinstance(data.tensor, torch.Tensor)
        assert data.tensor.shape == (2, 2)
        assert data.source == str(csv_file)

    def test_data_stores_source_attribute(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n2\n3")
        config = _make_config(str(csv_file))

        data = Data(config)

        assert data.source == str(csv_file)

    def test_data_raises_if_source_is_none(self) -> None:
        config = _make_config("")

        with pytest.raises(ValueError, match="No data source specified"):
            Data(config)

    def test_data_raises_if_source_does_not_exist(self) -> None:
        config = _make_config("/nonexistent/path/data.csv")

        with pytest.raises(ValueError, match="does not exist"):
            Data(config)

    def test_data_loads_filename_aligned_labels(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "ISIC_0002.jpg")
        _write_image(data_dir / "ISIC_0001.jpg")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("image,MEL,NV\nISIC_0002,1,0\nISIC_0001,0,1\n")
        config = _make_config(
            str(data_dir),
            labels_source=str(labels_file),
            labels_id_column="image",
        )

        data = Data(config)

        assert data.labels is not None
        assert data.labels.tolist() == [1, 0]

    def test_data_warns_and_uses_row_order_without_id_column(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "ISIC_0001.jpg")
        _write_image(data_dir / "ISIC_0002.jpg")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("label\n1\n0")
        config = _make_config(
            str(data_dir),
            labels_source=str(labels_file),
            labels_column="label",
        )

        with pytest.warns(UserWarning, match="labels id column"):
            data = Data(config)

        assert data.labels is not None
        assert data.labels.tolist() == [1, 0]

    def test_data_warns_and_drops_labels_if_filenames_missing(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "ISIC_0001.jpg")
        _write_image(data_dir / "ISIC_0002.jpg")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("image,label\nISIC_0001,0\n")
        config = _make_config(
            str(data_dir),
            labels_source=str(labels_file),
            labels_id_column="image",
            labels_column="label",
        )

        with pytest.warns(UserWarning, match="Missing labels"):
            data = Data(config)

        assert data.labels is None

    def test_data_aligns_ids_when_labels_include_extension(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "ISIC_0001.jpg")
        _write_image(data_dir / "ISIC_0002.jpg")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("image,label\nISIC_0001.jpg,1\nISIC_0002.jpg,0\n")
        config = _make_config(
            str(data_dir),
            labels_source=str(labels_file),
            labels_id_column="image",
            labels_column="label",
            labels_encoding="index",
        )

        data = Data(config)

        assert data.labels is not None
        assert data.labels.tolist() == [1, 0]

    def test_data_warns_and_drops_labels_for_duplicate_label_ids(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "ISIC_0001.jpg")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("image,label\nISIC_0001,1\nISIC_0001,0\n")
        config = _make_config(
            str(data_dir),
            labels_source=str(labels_file),
            labels_id_column="image",
            labels_column="label",
        )

        with pytest.warns(UserWarning, match="Duplicate label IDs"):
            data = Data(config)

        assert data.labels is None

    def test_data_raises_for_unsupported_labels_encoding(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n2")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("label\n0\n1")
        config = _make_config(
            str(csv_file),
            labels_source=str(labels_file),
            labels_column="label",
            labels_encoding="ordinal",
        )

        with pytest.raises(ValueError, match="Unsupported data.labels.encoding"):
            Data(config)


class TestDataDescribe:
    def test_describe_returns_dataset_metadata(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12")
        config = _make_config(str(csv_file), name="test_dataset")

        data = Data(config)
        info = data.describe()

        assert info["name"] == "test_dataset"
        assert info["source"] == str(csv_file)
        assert info["num_samples"] == 4
        assert info["shape"] == [4, 3]
        assert info["sample_shape"] == [3]
        assert info["dtype"] == "torch.float32"

    def test_describe_includes_sample_shape_for_multidimensional_data(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c,d,e\n1,2,3,4,5\n6,7,8,9,10")
        config = _make_config(str(csv_file))

        data = Data(config)
        info = data.describe()

        assert "sample_shape" in info
        assert info["sample_shape"] == [5]

    def test_describe_includes_sample_shape_for_2d_tensors(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n2\n3\n4\n5")
        config = _make_config(str(csv_file))

        data = Data(config)
        info = data.describe()

        assert info["num_samples"] == 5
        assert info["shape"] == [5, 1]
        assert "sample_shape" in info
        assert info["sample_shape"] == [1]


class TestDataLog:
    def test_log_calls_tracker_log_dataset(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4")
        config = _make_config(str(csv_file), name="test_data")
        data = Data(config)

        tracker = MagicMock()
        data.log(tracker)

        tracker.log_dataset.assert_called_once()
        call_args = tracker.log_dataset.call_args[0][0]
        assert call_args["source"] == str(csv_file)
        assert call_args["num_samples"] == 2
        assert call_args["shape"] == [2, 2]

    def test_log_includes_full_metadata(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9")
        config = _make_config(str(csv_file), name="my_dataset")
        data = Data(config)

        tracker = MagicMock()
        data.log(tracker)

        call_args = tracker.log_dataset.call_args[0][0]
        assert call_args["name"] == "my_dataset"
        assert "source" in call_args
        assert "num_samples" in call_args
        assert "shape" in call_args
        assert "dtype" in call_args
