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

from raitap.configs.schema import DirectoryLabelsConfig, TabularLabelsConfig
from raitap.data import Data
from raitap.data.types import InputModality


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
    labels_id_strategy: str | None = None,
) -> AppConfig:
    from raitap.data.types import IdStrategy, LabelEncoding

    if labels_source is not None:
        encoding = LabelEncoding(labels_encoding) if labels_encoding else None
        id_strategy = IdStrategy(labels_id_strategy) if labels_id_strategy else IdStrategy.auto
        labels = TabularLabelsConfig(
            source=labels_source,
            id_column=labels_id_column,
            column=labels_column,
            encoding=encoding,
            id_strategy=id_strategy,
        )
    else:
        labels = None

    return cast(
        "AppConfig",
        SimpleNamespace(
            data=SimpleNamespace(
                source=source,
                name=name,
                labels=labels,
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

        with pytest.raises(ValueError, match="could not be resolved"):
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
        assert isinstance(data.labels, torch.Tensor)
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
        assert isinstance(data.labels, torch.Tensor)
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
        assert isinstance(data.labels, torch.Tensor)
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

    def test_data_loads_nested_layout_with_relative_paths(self, tmp_path: Path) -> None:
        # Colliding stems across class subdirs — old stem-only matching would
        # silently drop rows; new id_strategy=auto resolves them as paths.
        data_dir = tmp_path / "images"
        (data_dir / "NORMAL").mkdir(parents=True)
        (data_dir / "PNEUMONIA").mkdir(parents=True)
        _write_image(data_dir / "NORMAL" / "IM-0001.jpeg")
        _write_image(data_dir / "PNEUMONIA" / "IM-0001.jpeg")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("image,label\nNORMAL/IM-0001.jpeg,0\nPNEUMONIA/IM-0001.jpeg,1\n")
        config = _make_config(
            str(data_dir),
            labels_source=str(labels_file),
            labels_id_column="image",
            labels_column="label",
            labels_encoding="index",
        )

        data = Data(config)

        assert isinstance(data.tensor, torch.Tensor)
        assert data.tensor.shape[0] == 2
        assert data.labels is not None
        # Sample order is sorted by relative posix path: NORMAL/* < PNEUMONIA/*.
        assert isinstance(data.labels, torch.Tensor)
        assert data.labels.tolist() == [0, 1]

    def test_data_auto_detects_relative_paths_from_separators(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "images"
        (data_dir / "a").mkdir(parents=True)
        (data_dir / "b").mkdir(parents=True)
        _write_image(data_dir / "a" / "x.jpg")
        _write_image(data_dir / "b" / "x.jpg")
        labels_file = tmp_path / "labels.csv"
        # Backslash separator should also trigger relative_path mode.
        labels_file.write_text("image,label\na\\x.jpg,7\nb\\x.jpg,8\n")
        config = _make_config(
            str(data_dir),
            labels_source=str(labels_file),
            labels_id_column="image",
            labels_column="label",
            labels_encoding="index",
        )

        data = Data(config)

        assert data.labels is not None
        assert isinstance(data.labels, torch.Tensor)
        assert data.labels.tolist() == [7, 8]

    def test_data_explicit_stem_strategy_with_nested_collisions_warns(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "images"
        (data_dir / "NORMAL").mkdir(parents=True)
        (data_dir / "PNEUMONIA").mkdir(parents=True)
        _write_image(data_dir / "NORMAL" / "IM-0001.jpeg")
        _write_image(data_dir / "PNEUMONIA" / "IM-0001.jpeg")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("image,label\nNORMAL/IM-0001.jpeg,0\nPNEUMONIA/IM-0001.jpeg,1\n")
        config = _make_config(
            str(data_dir),
            labels_source=str(labels_file),
            labels_id_column="image",
            labels_column="label",
            labels_encoding="index",
            labels_id_strategy="stem",
        )

        # Forcing stem mode collapses both rows to the same key → duplicate.
        with pytest.warns(UserWarning) as record:
            data = Data(config)
        assert data.labels is None
        msgs = [str(w.message) for w in record]
        assert any("Duplicate label IDs" in m for m in msgs)
        # Hint pins the specific guidance: switch strategy + use relative paths.
        assert any(
            "id_strategy=relative_path" in m and "stem-only matching collapses" in m for m in msgs
        )

    def test_data_missing_label_hint_for_nested_samples_flat_labels(self, tmp_path: Path) -> None:
        # Nested data + flat label ids under relative_path → strategy mismatch.
        data_dir = tmp_path / "images"
        (data_dir / "NORMAL").mkdir(parents=True)
        _write_image(data_dir / "NORMAL" / "IM-0001.jpeg")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("image,label\nIM-0001.jpeg,0\n")
        config = _make_config(
            str(data_dir),
            labels_source=str(labels_file),
            labels_id_column="image",
            labels_column="label",
            labels_encoding="index",
            labels_id_strategy="relative_path",
        )

        with pytest.warns(UserWarning) as record:
            data = Data(config)
        assert data.labels is None
        msgs = [str(w.message) for w in record]
        # Hint pins the specific direction: nested data, flat labels.
        assert any("Missing labels" in m and "nested layout" in m for m in msgs)

    def test_data_missing_label_hint_for_nested_labels_flat_samples(self, tmp_path: Path) -> None:
        # Inverse direction: flat data dir + label ids carrying directory
        # prefixes. Exercises the second relative_path hint branch.
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "Y.jpeg")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("image,label\nPNEUMONIA/Y.jpeg,0\n")
        config = _make_config(
            str(data_dir),
            labels_source=str(labels_file),
            labels_id_column="image",
            labels_column="label",
            labels_encoding="index",
            labels_id_strategy="relative_path",
        )

        with pytest.warns(UserWarning) as record:
            data = Data(config)
        assert data.labels is None
        msgs = [str(w.message) for w in record]
        assert any("Missing labels" in m and "label ids contain path separators" in m for m in msgs)

    def test_data_missing_label_no_hint_under_stem_mode(self, tmp_path: Path) -> None:
        # Stem mode strips dirs symmetrically, so a missing match is a real
        # basename gap — not a strategy issue. Hint must not fire for any
        # combination of separators (incl. asymmetric), since switching to
        # relative_path won't fix the gap either.
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "X.jpeg")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("image,label\nNORMAL/Y.jpeg,0\nPNEUMONIA/Z.jpeg,1\n")
        config = _make_config(
            str(data_dir),
            labels_source=str(labels_file),
            labels_id_column="image",
            labels_column="label",
            labels_encoding="index",
            labels_id_strategy="stem",
        )

        with pytest.warns(UserWarning) as record:
            data = Data(config)
        assert data.labels is None
        msgs = [str(w.message) for w in record]
        assert any("Missing labels" in m for m in msgs)
        # No strategy hint — switching strategies wouldn't make X match Y/Z.
        assert not any("id_strategy=relative_path" in m for m in msgs)
        assert not any("id_strategy=stem" in m and "Hint:" in m for m in msgs)

    def test_data_raises_for_unsupported_id_strategy(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        _write_image(data_dir / "x.jpg")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("image,label\nx,0\n")
        with pytest.raises(ValueError):
            config = _make_config(
                str(data_dir),
                labels_source=str(labels_file),
                labels_id_column="image",
                labels_column="label",
                labels_encoding="index",
                labels_id_strategy="bogus",
            )
            Data(config)

    def test_data_records_image_modality_for_image_dir(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        _write_image(img_dir / "a.png")
        _write_image(img_dir / "b.png")
        config = _make_config(str(img_dir))

        data = Data(config)

        assert data.input_modality is InputModality.image

    def test_data_records_tabular_modality_for_csv(self, tmp_path: Path) -> None:
        csv = tmp_path / "rows.csv"
        csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        config = _make_config(str(csv))

        data = Data(config)

        assert data.input_modality is InputModality.tabular

    def test_data_records_image_modality_for_demo_sample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exercise the SAMPLE_SOURCES branch hermetically: stub the networked loader.
        def _fake_load_sample(
            name: str, *, per_image_transform: object = None
        ) -> tuple[torch.Tensor, list[str]]:
            assert name == "imagenet_samples"
            assert per_image_transform is None or callable(per_image_transform)
            return torch.zeros(2, 3, 8, 8), ["x.jpg", "y.jpg"]

        monkeypatch.setattr("raitap.data.data._load_sample", _fake_load_sample)
        config = _make_config("imagenet_samples")

        data = Data(config)

        assert data.input_modality is InputModality.image

    def test_data_raises_for_unsupported_labels_encoding(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n2")
        labels_file = tmp_path / "labels.csv"
        labels_file.write_text("label\n0\n1")
        with pytest.raises(ValueError):
            config = _make_config(
                str(csv_file),
                labels_source=str(labels_file),
                labels_column="label",
                labels_encoding="ordinal",
            )
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


class TestLoadDirectoryLabelsViaParser:
    """Directory label behavior via DirectoryLabelParser (replaces deleted _load_directory_labels).

    The private _load_directory_labels function and load_classification_labels were removed in
    the discriminated-config refactor. Behavior is now covered by DirectoryLabelParser
    and _resolve_and_parse_labels. These tests preserve the behavioral contracts.
    """

    def _run_directory_parser(self, sample_ids: list[str] | None) -> torch.Tensor | None:
        from types import SimpleNamespace
        from typing import cast

        from raitap.data.data import _resolve_and_parse_labels
        from raitap.types import TaskKind

        cfg = cast(
            "AppConfig",
            SimpleNamespace(
                data=SimpleNamespace(labels=DirectoryLabelsConfig(), source=None),
                model=SimpleNamespace(class_names=None),
            ),
        )
        return _resolve_and_parse_labels(
            cfg, task_kind=TaskKind.classification, tensor=None, sample_ids=sample_ids
        )

    def test_derives_labels_from_top_level_class_folder(self) -> None:
        result = self._run_directory_parser(["NORMAL/a.jpg", "PNEUMONIA/b.jpg", "NORMAL/c.jpg"])
        assert result is not None
        assert torch.equal(result, torch.tensor([0, 1, 0]))

    def test_nesting_within_class_stays_top_level(self) -> None:
        result = self._run_directory_parser(["NORMAL/sub/a.jpg", "PNEUMONIA/b.jpg"])
        assert result is not None
        assert torch.equal(result, torch.tensor([0, 1]))

    def test_single_class_is_all_zeros_not_error(self) -> None:
        result = self._run_directory_parser(["NORMAL/a.jpg", "NORMAL/b.jpg"])
        assert result is not None
        assert torch.equal(result, torch.tensor([0, 0]))

    def test_sample_without_class_subdir_returns_none(self) -> None:
        result = self._run_directory_parser(["a.jpg", "NORMAL/b.jpg"])
        assert result is None

    def test_none_sample_ids_returns_none(self) -> None:
        result = self._run_directory_parser(None)
        assert result is None

    def test_empty_sample_ids_returns_none(self) -> None:
        result = self._run_directory_parser([])
        assert result is None

    def test_directory_source_derives_labels_from_layout(self, tmp_path: Path) -> None:
        """Data with DirectoryLabelsConfig derives labels from the sample layout."""
        from types import SimpleNamespace
        from typing import cast


        img_dir = tmp_path / "images"
        (img_dir / "NORMAL").mkdir(parents=True)
        (img_dir / "PNEUMONIA").mkdir(parents=True)
        _write_image(img_dir / "NORMAL" / "a.jpg")
        _write_image(img_dir / "PNEUMONIA" / "b.jpg")
        _write_image(img_dir / "NORMAL" / "c.jpg")

        cfg = cast(
            "AppConfig",
            SimpleNamespace(
                data=SimpleNamespace(
                    source=str(img_dir),
                    name="test_dir",
                    labels=DirectoryLabelsConfig(),
                )
            ),
        )
        data = Data(cfg)

        assert data.labels is not None
        assert isinstance(data.labels, torch.Tensor)
        # NORMAL=0, PNEUMONIA=1; sorted by posix path: NORMAL/a, NORMAL/c, PNEUMONIA/b
        assert data.labels.tolist() == [0, 0, 1]
