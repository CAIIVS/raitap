"""Unit tests for the Data class."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

from raitap.data import Data


def _make_config(source: str, name: str = "test_data") -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            data=SimpleNamespace(
                source=source,
                name=name,
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


class TestDataDescribe:
    def test_describe_returns_dataset_metadata(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12")
        config = _make_config(str(csv_file), name="test_dataset")

        data = Data(config)
        info = data.describe()

        assert info["name"] == "dataset"
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

    def test_describe_omits_sample_shape_for_1d_tensors(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n2\n3\n4\n5")
        config = _make_config(str(csv_file))

        data = Data(config)
        info = data.describe()

        assert info["num_samples"] == 5
        assert info["shape"] == [5, 1]
        assert "sample_shape" in info


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
        assert "name" in call_args
        assert "source" in call_args
        assert "num_samples" in call_args
        assert "shape" in call_args
        assert "dtype" in call_args
