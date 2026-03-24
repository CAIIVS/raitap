"""Tests for the raitap.data module (co-located)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

from raitap.data import Data, get_source_path
from raitap.data.data import _load_images, _load_tabular, _load_tabular_dir
from raitap.data.samples import SAMPLE_SOURCES, _load_sample, _resolve_sample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_image(path: Path, width: int = 32, height: int = 32) -> None:
    """Write a small solid-colour RGB JPEG to *path*."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _write_csv(path: Path, rows: int = 4, cols: int = 5) -> None:
    """Write a minimal float CSV to *path*."""
    header = ",".join(f"f{i}" for i in range(cols))
    data_rows = "\n".join(
        ",".join(str(float(r * cols + c)) for c in range(cols)) for r in range(rows)
    )
    path.write_text(f"{header}\n{data_rows}")


# ---------------------------------------------------------------------------
# get_source_path
# ---------------------------------------------------------------------------


class TestGetSourcePath:
    def test_existing_local_path_returned(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2")
        result = get_source_path(str(f))
        assert result == f

    def test_unknown_source_raises(self) -> None:
        with pytest.raises(ValueError, match="could not be resolved"):
            get_source_path("/totally/nonexistent/path/data.csv")

    def test_url_downloads_and_caches(self, tmp_path: Path) -> None:
        fake_content = b"fake file content"
        mock_response = MagicMock()
        mock_response.read.return_value = fake_content
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch("raitap.data.data.urllib.request.urlopen", return_value=mock_response),
            patch("raitap.data.data._CACHE_DIR", tmp_path),
        ):
            result = get_source_path("https://example.com/file.csv")

        assert result.exists()
        assert result.read_bytes() == fake_content

    def test_url_skips_download_if_cached(self, tmp_path: Path) -> None:
        dest = tmp_path / "downloads" / "file.csv"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"cached")

        with (
            patch("raitap.data.data.urllib.request.urlopen") as mock_open,
            patch("raitap.data.data._CACHE_DIR", tmp_path),
        ):
            result = get_source_path("https://example.com/file.csv")

        mock_open.assert_not_called()
        assert result == dest


# ---------------------------------------------------------------------------
# _load_images
# ---------------------------------------------------------------------------


class TestLoadImages:
    def test_single_image_file(self, tmp_path: Path) -> None:
        img_path = tmp_path / "img.jpg"
        _write_image(img_path)
        tensor = _load_images(img_path)
        assert tensor.shape == (1, 3, 32, 32)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0

    def test_directory_of_images(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_image(tmp_path / f"img{i}.jpg")
        tensor = _load_images(tmp_path)
        assert tensor.shape == (3, 3, 32, 32)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No image files found"):
            _load_images(tmp_path)

    def test_inconsistent_sizes_raise(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "a.jpg", 32, 32)
        _write_image(tmp_path / "b.jpg", 64, 64)
        with pytest.raises(ValueError, match="inconsistent shapes"):
            _load_images(tmp_path)

    def test_non_image_files_ignored(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "img.png")
        (tmp_path / "readme.txt").write_text("ignore me")
        tensor = _load_images(tmp_path)
        assert tensor.shape[0] == 1


# ---------------------------------------------------------------------------
# _load_tabular / _load_tabular_dir
# ---------------------------------------------------------------------------


class TestLoadTabular:
    def test_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        _write_csv(p, rows=4, cols=5)
        tensor = _load_tabular(p)
        assert tensor.shape == (4, 5)
        assert tensor.dtype == torch.float32

    def test_tsv(self, tmp_path: Path) -> None:
        p = tmp_path / "data.tsv"
        p.write_text("a\tb\n1.0\t2.0\n3.0\t4.0")
        tensor = _load_tabular(p)
        assert tensor.shape == (2, 2)

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported tabular format"):
            _load_tabular(p)


class TestLoadTabularDir:
    def test_multiple_csvs_concatenated(self, tmp_path: Path) -> None:
        _write_csv(tmp_path / "a.csv", rows=3, cols=4)
        _write_csv(tmp_path / "b.csv", rows=5, cols=4)
        tensor = _load_tabular_dir(tmp_path)
        assert tensor.shape == (8, 4)

    def test_inconsistent_columns_raise(self, tmp_path: Path) -> None:
        _write_csv(tmp_path / "a.csv", rows=2, cols=3)
        _write_csv(tmp_path / "b.csv", rows=2, cols=5)
        with pytest.raises(ValueError, match="inconsistent column counts"):
            _load_tabular_dir(tmp_path)


# ---------------------------------------------------------------------------
# Data class  (integration-style, local files only)
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_local_image_file(self, tmp_path: Path) -> None:
        p = tmp_path / "img.png"
        _write_image(p)
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"data": type("DataConfig", (), {"source": str(p)})})(),
        )
        data = Data(cfg)
        assert data.tensor.shape == (1, 3, 32, 32)

    def test_local_image_directory(self, tmp_path: Path) -> None:
        for i in range(2):
            _write_image(tmp_path / f"img{i}.jpg")
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"data": type("DataConfig", (), {"source": str(tmp_path)})})(),
        )
        data = Data(cfg)
        assert data.tensor.shape[0] == 2

    def test_local_csv_file(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        _write_csv(p, rows=6, cols=3)
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"data": type("DataConfig", (), {"source": str(p)})})(),
        )
        data = Data(cfg)
        assert data.tensor.shape == (6, 3)

    def test_local_tabular_directory(self, tmp_path: Path) -> None:
        _write_csv(tmp_path / "a.csv", rows=2, cols=3)
        _write_csv(tmp_path / "b.csv", rows=4, cols=3)
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"data": type("DataConfig", (), {"source": str(tmp_path)})})(),
        )
        data = Data(cfg)
        assert data.tensor.shape == (6, 3)

    def test_mixed_directory_raises(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "img.jpg")
        _write_csv(tmp_path / "data.csv")
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"data": type("DataConfig", (), {"source": str(tmp_path)})})(),
        )
        with pytest.raises(ValueError, match="both image and tabular"):
            Data(cfg)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"data": type("DataConfig", (), {"source": str(tmp_path)})})(),
        )
        with pytest.raises(FileNotFoundError, match="No supported files"):
            Data(cfg)

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "data.xyz"
        p.write_text("something")
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"data": type("DataConfig", (), {"source": str(p)})})(),
        )
        with pytest.raises(ValueError, match="Cannot infer data type"):
            Data(cfg)

    def test_invalid_source_raises(self) -> None:
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"data": type("DataConfig", (), {"source": "/no/such/path/file.csv"})},
            )(),
        )
        with pytest.raises(ValueError, match="does not exist"):
            Data(cfg)


class TestDescribeData:
    def test_describe_data_includes_shape_dtype_and_sample_shape(self, tmp_path: Path) -> None:
        img_file = tmp_path / "img.jpg"
        _write_image(img_file, 32, 32)
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {
                    "data": type(
                        "DataConfig", (), {"source": str(img_file), "name": "imagenet_samples"}
                    )
                },
            )(),
        )
        data = Data(cfg)
        data.source = "/tmp/imagenet"

        info = data.describe()

        assert info["name"] == "dataset"
        assert info["source"] == "/tmp/imagenet"
        assert info["num_samples"] == 1
        assert info["shape"] == [1, 3, 32, 32]
        assert info["sample_shape"] == [3, 32, 32]
        assert info["dtype"] == "torch.float32"

    def test_describe_data_without_sample_shape_for_1d_input(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n2\n3\n4\n5\n6\n7\n8")
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"data": type("DataConfig", (), {"source": str(csv_file), "name": "vector_data"})},
            )(),
        )
        data = Data(cfg)

        info = data.describe()

        assert info["name"] == "dataset"
        assert info["num_samples"] == 8
        assert info["shape"] == [8, 1]
        assert info["dtype"] == "torch.float32"


# ---------------------------------------------------------------------------
# samples module
# ---------------------------------------------------------------------------


class TestResolveSample:
    def test_unknown_name_returns_none(self) -> None:
        assert _resolve_sample("not_a_real_dataset") is None

    def test_non_string_returns_none(self) -> None:
        assert _resolve_sample(42) is None  # type: ignore[arg-type]

    def test_known_name_creates_directory_and_returns_path(self, tmp_path: Path) -> None:
        name = next(iter(SAMPLE_SOURCES))
        mock_response = MagicMock()
        mock_response.read.return_value = b"\xff\xd8\xff\xe0"  # minimal bytes
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch("raitap.data.samples._CACHE_DIR", tmp_path),
            patch("raitap.data.samples.urllib.request.urlopen", return_value=mock_response),
        ):
            result = _resolve_sample(name)

        assert result is not None
        assert result.is_dir()

    def test_known_name_skips_existing_files(self, tmp_path: Path) -> None:
        name = next(iter(SAMPLE_SOURCES))
        cache_dir = tmp_path / name
        cache_dir.mkdir(parents=True)
        for _, filename in SAMPLE_SOURCES[name]:
            (cache_dir / filename).write_bytes(b"cached")

        with (
            patch("raitap.data.samples._CACHE_DIR", tmp_path),
            patch("raitap.data.samples.urllib.request.urlopen") as mock_open,
        ):
            _resolve_sample(name)

        mock_open.assert_not_called()


class TestSampleSources:
    def test_all_known_samples_have_entries(self) -> None:
        assert len(SAMPLE_SOURCES) > 0
        for name, entries in SAMPLE_SOURCES.items():
            assert isinstance(name, str)
            assert len(entries) > 0
            for url, filename in entries:
                assert url.startswith("http")
                assert filename

    def test_load_sample_with_real_images(self, tmp_path: Path) -> None:
        name = next(iter(SAMPLE_SOURCES))
        cache_dir = tmp_path / name
        cache_dir.mkdir(parents=True)
        for _, filename in SAMPLE_SOURCES[name]:
            _write_image(cache_dir / filename, 64, 64)

        with patch("raitap.data.samples._CACHE_DIR", tmp_path):
            tensor = _load_sample(name, size=32)

        n = len(SAMPLE_SOURCES[name])
        assert tensor.shape == (n, 3, 32, 32)
        assert tensor.dtype == torch.float32

    def test_load_sample_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not a known demo sample"):
            _load_sample("nonexistent_dataset")
