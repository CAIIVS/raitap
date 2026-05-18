from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

from raitap.data import Data
from raitap.data.data import (
    SourceKind,
    _load_images,
    _load_tabular,
    _load_tabular_dir,
    get_source_path,
    load_numpy_from_source,
    load_tensor_from_source,
)
from raitap.data.samples import SAMPLE_SOURCES, _load_sample, _resolve_sample
from raitap.types import Hardware

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

    def test_sample_name_resolves_to_cache_dir(self, tmp_path: Path) -> None:
        with (
            patch("raitap.data.samples._CACHE_DIR", tmp_path),
            patch("raitap.data.samples.download_file") as mock_download,
        ):

            def _create_file(_url: str, dest: Path) -> None:
                dest.write_bytes(b"x")

            mock_download.side_effect = _create_file
            result = get_source_path("imagenet_samples")

        assert result.is_dir()
        assert result == tmp_path / "imagenet_samples"

    def test_sample_name_labels_resolves_to_csv(self, tmp_path: Path) -> None:
        with (
            patch("raitap.data.samples._CACHE_DIR", tmp_path),
            patch("raitap.data.samples.download_file") as mock_download,
        ):

            def _create_file(_url: str, dest: Path) -> None:
                dest.write_bytes(b"x")

            mock_download.side_effect = _create_file
            result = get_source_path("imagenet_samples", kind=SourceKind.LABELS)

        assert result.is_file()
        assert result.name == "labels.csv"
        content = result.read_text(encoding="utf-8")
        assert "image,label" in content
        assert "tench.jpg,0" in content
        assert "golden_retriever.jpg,207" in content

    def test_sample_name_without_labels_raises(self, tmp_path: Path) -> None:
        with (
            patch("raitap.data.samples._CACHE_DIR", tmp_path),
            patch("raitap.data.samples.download_file") as mock_download,
        ):
            mock_download.side_effect = lambda _url, dest: dest.write_bytes(b"x")
            with pytest.raises(ValueError, match="does not ship ground-truth labels"):
                get_source_path("malaria", kind=SourceKind.LABELS)

    def test_invalid_kind_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2")
        with pytest.raises(ValueError, match="Invalid kind"):
            get_source_path(str(f), kind=cast("Any", "lables"))

    def test_url_downloads_and_caches(self, tmp_path: Path) -> None:
        fake_content = b"fake file content"
        mock_response = MagicMock()
        mock_response.read.return_value = fake_content
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch("raitap.data.utils.urllib.request.urlopen", return_value=mock_response),
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
            patch("raitap.data.utils.urllib.request.urlopen") as mock_open,
            patch("raitap.data.data._CACHE_DIR", tmp_path),
        ):
            result = get_source_path("https://example.com/file.csv")

        mock_open.assert_not_called()
        assert result == dest

    def test_url_download_failure_propagates(self, tmp_path: Path) -> None:
        with (
            patch("raitap.data.data._CACHE_DIR", tmp_path),
            patch("raitap.data.data.download_file", side_effect=OSError("network down")),
            pytest.raises(OSError, match="network down"),
        ):
            get_source_path("https://example.com/file.csv")


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


class TestDataPreprocessing:
    """``Data._load_data`` resolves preprocessing and applies data preprocessing
    per-image before stacking, so mixed-size directories load successfully
    when ``data.preprocessing: model-bundled`` is set."""

    @staticmethod
    def _make_cfg(source: str, *, preprocessing: str | None) -> AppConfig:
        from raitap.configs.schema import AppConfig, DataConfig, LabelsConfig, ModelConfig

        return cast(
            "AppConfig",
            AppConfig(
                model=ModelConfig(source="resnet50"),
                data=DataConfig(
                    name="test",
                    source=source,
                    preprocessing=preprocessing,
                    labels=LabelsConfig(),
                ),
                hardware=Hardware.cpu,
            ),
        )

    def test_mixed_size_dir_with_model_bundled_loads(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "a.jpg", 451, 800)
        _write_image(tmp_path / "b.jpg", 533, 800)
        _write_image(tmp_path / "c.jpg", 440, 780)
        cfg = self._make_cfg(str(tmp_path), preprocessing="model-bundled")
        data = Data(cfg)
        assert data.tensor.shape == (3, 3, 224, 224)
        assert data.tensor.dtype == torch.float32

    def test_mixed_size_dir_without_preprocessing_raises(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "a.jpg", 451, 800)
        _write_image(tmp_path / "b.jpg", 533, 800)
        cfg = self._make_cfg(str(tmp_path), preprocessing=None)
        with pytest.raises(ValueError, match="inconsistent shapes"):
            Data(cfg)

    def test_uniform_dir_without_preprocessing_still_loads(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_image(tmp_path / f"img{i}.jpg", 64, 64)
        cfg = self._make_cfg(str(tmp_path), preprocessing=None)
        data = Data(cfg)
        assert data.tensor.shape == (3, 3, 64, 64)

    def test_supplied_resolved_preprocessing_skips_resolution(self, tmp_path: Path) -> None:
        from torch import nn

        from raitap.configs.schema import AppConfig, DataConfig, LabelsConfig, ModelConfig
        from raitap.data.preprocessing import ResolvedPreprocessing

        class _ShapeModule(nn.Module):
            def forward(self, image: torch.Tensor) -> torch.Tensor:
                return torch.zeros(3, 8, 8, dtype=image.dtype)

        _write_image(tmp_path / "a.jpg", 32, 32)
        cfg = cast(
            "AppConfig",
            AppConfig(
                model=ModelConfig(source="resnet50"),
                data=DataConfig(
                    name="test",
                    source=str(tmp_path),
                    preprocessing="model-bundled",
                    labels=LabelsConfig(),
                ),
                hardware=Hardware.cpu,
            ),
        )
        resolved = ResolvedPreprocessing(
            data_module=_ShapeModule(),
            model_module=None,
            data_origin="model-bundled",
            model_origin="off",
            description="supplied",
        )

        with patch("raitap.data.data.resolve_preprocessing") as resolve_preprocessing_mock:
            resolve_preprocessing_mock.side_effect = AssertionError("should not resolve again")
            data = Data(cfg, resolved_preprocessing=resolved)

        assert data.tensor.shape == (1, 3, 8, 8)
        resolve_preprocessing_mock.assert_not_called()

    def test_onnx_custom_file_data_factory_drives_data_loading(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from raitap.configs.schema import AppConfig, DataConfig, LabelsConfig, ModelConfig

        _write_image(tmp_path / "a.jpg", 32, 48)
        _write_image(tmp_path / "b.jpg", 40, 64)
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx")
        preprocessing_path = tmp_path / "preprocessing.py"
        preprocessing_path.write_text(
            "import torch\n"
            "from torch import nn\n"
            "from raitap.data import (\n"
            "    raitap_model_input_transformation_factory,\n"
            "    raitap_preprocessing_factory,\n"
            ")\n"
            "\n"
            "class _ModelOnly(nn.Module):\n"
            "    def forward(self, image: torch.Tensor) -> torch.Tensor:\n"
            "        raise AssertionError('model input transformation should stay backend-side')\n"
            "\n"
            "class _DataOnly(nn.Module):\n"
            "    def forward(self, image: torch.Tensor) -> torch.Tensor:\n"
            "        return torch.zeros(3, 8, 8, dtype=image.dtype)\n"
            "\n"
            "@raitap_model_input_transformation_factory\n"
            "def model_input_transform() -> nn.Module:\n"
            "    return _ModelOnly()\n"
            "\n"
            "@raitap_preprocessing_factory\n"
            "def data_preprocessing() -> nn.Module:\n"
            "    return _DataOnly()\n"
        )
        monkeypatch.setenv("RAITAP_ALLOW_PREPROCESSING_EXEC", "1")
        cfg = cast(
            "AppConfig",
            AppConfig(
                model=ModelConfig(source=str(model_path)),
                data=DataConfig(
                    name="test",
                    source=str(tmp_path),
                    preprocessing=str(preprocessing_path),
                    labels=LabelsConfig(),
                ),
                hardware=Hardware.cpu,
            ),
        )

        data = Data(cfg)

        assert data.tensor.shape == (2, 3, 8, 8)
        assert data.tensor.dtype == torch.float32

    def test_load_tensor_from_source_applies_per_image_transform(self, tmp_path: Path) -> None:
        """``load_tensor_from_source`` (used by ``resolve_call_data_sources``
        for SHAP background_data etc.) honours ``per_image_transform`` so
        mixed-size auxiliary directories load to a uniform shape."""
        from raitap.configs.adapter_factory import resolve_per_image_transform
        from raitap.data import load_tensor_from_source

        _write_image(tmp_path / "a.jpg", 451, 800)
        _write_image(tmp_path / "b.jpg", 533, 800)
        _write_image(tmp_path / "c.jpg", 440, 780)

        cfg = self._make_cfg(str(tmp_path), preprocessing="model-bundled")
        per_image_transform = resolve_per_image_transform(cfg)
        assert per_image_transform is not None

        tensor = load_tensor_from_source(str(tmp_path), per_image_transform=per_image_transform)
        assert tensor.shape == (3, 3, 224, 224)

    def test_load_tensor_from_source_without_transform_raises_on_mixed_sizes(
        self, tmp_path: Path
    ) -> None:
        """Regression: confirms the helper is the actual choke point when no
        transform is supplied — same failure mode as the primary loader."""
        from raitap.data import load_tensor_from_source

        _write_image(tmp_path / "a.jpg", 451, 800)
        _write_image(tmp_path / "b.jpg", 533, 800)
        with pytest.raises(ValueError, match="inconsistent shapes"):
            load_tensor_from_source(str(tmp_path))

    def test_sample_source_loads_native_resolution_then_transforms(self, tmp_path: Path) -> None:
        """Regression for the demo pre-resize bug: sample sources must load
        images at their native resolution and let ``per_image_transform``
        do the shape work. Otherwise the bundled Resize/CenterCrop sees an
        already-squashed image instead of the original — which silently
        breaks pretrained-weight accuracy on `raitap --demo`."""
        from torch import nn

        from raitap.configs.schema import AppConfig, DataConfig, LabelsConfig, ModelConfig
        from raitap.data.samples import SAMPLE_SOURCES

        # Stage a fake sample at varied native sizes so the test would fail
        # if the loader fell back to the legacy 224x224 PIL squash.
        sample_dir = tmp_path / "fake_native_samples"
        sample_dir.mkdir()
        for name, w, h in [("a.jpg", 451, 800), ("b.jpg", 533, 800), ("c.jpg", 440, 780)]:
            _write_image(sample_dir / name, w, h)

        original = SAMPLE_SOURCES.copy()
        SAMPLE_SOURCES["fake_native_samples"] = []
        try:
            calls: list[tuple[int, ...]] = []

            class _SpyModule(nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    calls.append(tuple(x.shape))
                    return torch.zeros(3, 224, 224)

            with (
                patch("raitap.data.samples._CACHE_DIR", tmp_path),
                patch("raitap.data.data.resolve_preprocessing") as resolve_preprocessing_mock,
            ):
                from raitap.data.preprocessing import ResolvedPreprocessing

                resolve_preprocessing_mock.return_value = ResolvedPreprocessing(
                    data_module=_SpyModule(),
                    model_module=None,
                    data_origin="model-bundled",
                    model_origin="off",
                    description="spy",
                )

                cfg = cast(
                    "AppConfig",
                    AppConfig(
                        model=ModelConfig(source="resnet50"),
                        data=DataConfig(
                            name="fake_native_samples",
                            source="fake_native_samples",
                            preprocessing="model-bundled",
                            labels=LabelsConfig(),
                        ),
                        hardware=Hardware.cpu,
                    ),
                )
                data = Data(cfg)
        finally:
            SAMPLE_SOURCES.clear()
            SAMPLE_SOURCES.update(original)

        assert data.tensor.shape == (3, 3, 224, 224)
        # The spy must see three per-image calls at NATIVE resolution
        # (the PIL pre-squash to 224x224 must NOT happen when a transform
        # is supplied). Shapes are (C, H, W) in PIL height/width order.
        assert len(calls) == 3
        native_shapes = {(3, 800, 451), (3, 800, 533), (3, 780, 440)}
        assert set(calls) == native_shapes


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

    def test_parquet(self, tmp_path: Path) -> None:
        p = tmp_path / "data.parquet"
        p.write_bytes(b"parquet-placeholder")
        parquet_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with patch("raitap.data.data.pd.read_parquet", return_value=parquet_df):
            tensor = _load_tabular(p)
        assert tensor.shape == (2, 2)
        assert tensor.dtype == torch.float32


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
            type(
                "AppConfig",
                (),
                {"data": type("DataConfig", (), {"source": str(p), "name": "isic2018"})},
            )(),
        )
        data = Data(cfg)
        assert data.tensor.shape == (1, 3, 32, 32)

    def test_local_image_directory(self, tmp_path: Path) -> None:
        for i in range(2):
            _write_image(tmp_path / f"img{i}.jpg")
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"data": type("DataConfig", (), {"source": str(tmp_path), "name": "isic2018"})},
            )(),
        )
        data = Data(cfg)
        assert data.tensor.shape[0] == 2

    def test_local_csv_file(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        _write_csv(p, rows=6, cols=3)
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"data": type("DataConfig", (), {"source": str(p), "name": "isic2018"})},
            )(),
        )
        data = Data(cfg)
        assert data.tensor.shape == (6, 3)

    def test_local_tabular_directory(self, tmp_path: Path) -> None:
        _write_csv(tmp_path / "a.csv", rows=2, cols=3)
        _write_csv(tmp_path / "b.csv", rows=4, cols=3)
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"data": type("DataConfig", (), {"source": str(tmp_path), "name": "isic2018"})},
            )(),
        )
        data = Data(cfg)
        assert data.tensor.shape == (6, 3)

    def test_mixed_directory_raises(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "img.jpg")
        _write_csv(tmp_path / "data.csv")
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"data": type("DataConfig", (), {"source": str(tmp_path), "name": "isic2018"})},
            )(),
        )
        with pytest.raises(ValueError, match="both image and tabular"):
            Data(cfg)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"data": type("DataConfig", (), {"source": str(tmp_path), "name": "isic2018"})},
            )(),
        )
        with pytest.raises(FileNotFoundError, match="No supported files"):
            Data(cfg)

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "data.xyz"
        p.write_text("something")
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"data": type("DataConfig", (), {"source": str(p), "name": "isic2018"})},
            )(),
        )
        with pytest.raises(ValueError, match="Cannot infer data type"):
            Data(cfg)

    def test_tabular_applies_data_module(self, tmp_path: Path) -> None:
        from torch import nn

        from raitap.configs.schema import AppConfig, DataConfig, LabelsConfig, ModelConfig
        from raitap.data.preprocessing import ResolvedPreprocessing

        class _ScaleModule(nn.Module):
            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                return batch * 10.0

        p = tmp_path / "rows.csv"
        _write_csv(p, rows=4, cols=3)
        cfg = cast(
            "AppConfig",
            AppConfig(
                model=ModelConfig(source="resnet50"),
                data=DataConfig(
                    name="tab",
                    source=str(p),
                    preprocessing="./scale.py",
                    labels=LabelsConfig(),
                ),
                hardware=Hardware.cpu,
            ),
        )
        resolved = ResolvedPreprocessing(
            data_module=_ScaleModule(),
            model_module=None,
            data_origin="custom-file",
            model_origin="off",
            description="supplied",
        )
        baseline = _load_tabular(p)

        data = Data(cfg, resolved_preprocessing=resolved)

        assert data.tensor.shape == baseline.shape
        torch.testing.assert_close(data.tensor, baseline * 10.0)

    def test_tabular_no_data_module_passes_through(self, tmp_path: Path) -> None:
        p = tmp_path / "rows.csv"
        _write_csv(p, rows=2, cols=3)
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"data": type("DataConfig", (), {"source": str(p), "name": "tab"})},
            )(),
        )
        data = Data(cfg)
        torch.testing.assert_close(data.tensor, _load_tabular(p))

    def test_invalid_source_raises(self) -> None:
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {
                    "data": type(
                        "DataConfig", (), {"source": "/no/such/path/file.csv", "name": "isic2018"}
                    )
                },
            )(),
        )
        with pytest.raises(ValueError, match="could not be resolved"):
            Data(cfg)

    def test_url_source_loads_csv_via_get_source_path(self, tmp_path: Path) -> None:
        p = tmp_path / "remote.csv"
        _write_csv(p, rows=2, cols=3)
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {
                    "data": type(
                        "DataConfig",
                        (),
                        {"source": "https://example.com/remote.csv", "name": "isic2018"},
                    )
                },
            )(),
        )
        with patch("raitap.data.data.get_source_path", return_value=p):
            data = Data(cfg)
        assert data.tensor.shape == (2, 3)

    def test_url_source_loads_image_via_get_source_path(self, tmp_path: Path) -> None:
        p = tmp_path / "remote.png"
        _write_image(p)
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {
                    "data": type(
                        "DataConfig",
                        (),
                        {"source": "https://example.com/remote.png", "name": "isic2018"},
                    )
                },
            )(),
        )
        with patch("raitap.data.data.get_source_path", return_value=p):
            data = Data(cfg)
        assert data.tensor.shape == (1, 3, 32, 32)

    def test_sample_labels_align_with_sample_images(self, tmp_path: Path) -> None:
        from raitap.data.samples import SAMPLE_LABELS

        with (
            patch("raitap.data.samples._CACHE_DIR", tmp_path),
            patch("raitap.data.samples.download_file") as mock_download,
            patch("raitap.data.samples._DEMO_SIZE", 32),
        ):
            mock_download.side_effect = lambda _url, dest: _write_image(dest, 32, 32)
            cfg = cast(
                "AppConfig",
                type(
                    "AppConfig",
                    (),
                    {
                        "data": type(
                            "DataConfig",
                            (),
                            {
                                "source": "imagenet_samples",
                                "name": "imagenet_samples",
                                "labels": type(
                                    "LabelsConfig",
                                    (),
                                    {
                                        "source": "imagenet_samples",
                                        "id_column": "image",
                                        "column": "label",
                                        "encoding": "index",
                                    },
                                )(),
                            },
                        )()
                    },
                )(),
            )
            data = Data(cfg)

        expected_ids = sorted(SAMPLE_LABELS["imagenet_samples"].keys())
        assert data.sample_ids == expected_ids
        expected_labels = [SAMPLE_LABELS["imagenet_samples"][fn] for fn in expected_ids]
        assert data.labels is not None
        assert data.labels.tolist() == expected_labels


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

        assert info["name"] == "imagenet_samples"
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

        assert info["name"] == "vector_data"
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
            patch("raitap.data.utils.urllib.request.urlopen", return_value=mock_response),
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
            patch("raitap.data.utils.urllib.request.urlopen") as mock_open,
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
            tensor, sample_ids = _load_sample(name, size=32)

        n = len(SAMPLE_SOURCES[name])
        assert tensor.shape == (n, 3, 32, 32)
        assert tensor.dtype == torch.float32
        assert len(sample_ids) == n
        assert sample_ids == sorted(sample_ids)

    def test_load_sample_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not a known demo sample"):
            _load_sample("nonexistent_dataset")


# ---------------------------------------------------------------------------
# load_tensor_from_source
# ---------------------------------------------------------------------------


class TestLoadTensorFromSource:
    def test_loads_from_local_image_dir(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(3):
            _write_image(img_dir / f"img{i}.png", width=8, height=8)

        tensor = load_tensor_from_source(str(img_dir))

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 3, 8, 8)
        assert tensor.dtype == torch.float32

    def test_loads_from_local_csv_file(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, rows=5, cols=4)

        tensor = load_tensor_from_source(str(csv_file))

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (5, 4)

    def test_n_samples_subsamples_rows(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, rows=20, cols=3)

        tensor = load_tensor_from_source(str(csv_file), n_samples=5)

        assert tensor.shape[0] == 5

    def test_n_samples_noop_when_smaller_than_dataset(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, rows=4, cols=3)

        tensor = load_tensor_from_source(str(csv_file), n_samples=100)

        assert tensor.shape[0] == 4

    def test_nonexistent_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="could not be resolved"):
            load_tensor_from_source("/nonexistent/path/data.csv")

    def test_unsupported_extension_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Cannot infer data type"):
            load_tensor_from_source(str(f))

    def test_named_demo_sample(self, tmp_path: Path) -> None:
        name = next(iter(SAMPLE_SOURCES))
        cache_dir = tmp_path / name
        cache_dir.mkdir(parents=True)
        for _, filename in SAMPLE_SOURCES[name]:
            _write_image(cache_dir / filename, 16, 16)

        with patch("raitap.data.samples._CACHE_DIR", tmp_path):
            tensor = load_tensor_from_source(name)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 4  # (N, C, H, W)


# ---------------------------------------------------------------------------
# load_numpy_from_source
# ---------------------------------------------------------------------------


class TestLoadNumpyFromSource:
    def test_matches_tensor_loader_for_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, rows=6, cols=3)

        tensor = load_tensor_from_source(str(csv_file))
        arr = load_numpy_from_source(str(csv_file))

        np.testing.assert_array_equal(arr, tensor.detach().cpu().numpy())

    def test_matches_tensor_loader_for_image_dir(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(2):
            _write_image(img_dir / f"img{i}.png", width=8, height=8)

        tensor = load_tensor_from_source(str(img_dir))
        arr = load_numpy_from_source(str(img_dir))

        np.testing.assert_array_equal(arr, tensor.detach().cpu().numpy())
