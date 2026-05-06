from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from raitap.data.metadata import (
    DataInputMetadata,
    infer_data_input_metadata,
    is_image_source,
    is_tabular_source,
    shape_tuple,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_infer_data_input_metadata_prefers_explicit_config_metadata() -> None:
    config = SimpleNamespace(
        data=SimpleNamespace(
            input_metadata={
                "kind": "text",
                "shape": [2, 5],
                "layout": "(B,T)",
                "feature_names": ["token_1", "token_2"],
                "tokenizer": "demo",
            }
        )
    )
    data = SimpleNamespace(tensor=SimpleNamespace(shape=(2, 3)), source="ignored.csv")

    metadata = infer_data_input_metadata(config, data)

    assert metadata == DataInputMetadata(
        kind="text",
        shape=(2, 5),
        layout="(B,T)",
        feature_names=["token_1", "token_2"],
        metadata={
            "kind": "text",
            "shape": [2, 5],
            "layout": "(B,T)",
            "feature_names": ["token_1", "token_2"],
            "tokenizer": "demo",
        },
    )


def test_infer_data_input_metadata_detects_image_source(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"not-an-image-for-this-test")
    config = SimpleNamespace(data=SimpleNamespace(source=str(image_path)))
    data = SimpleNamespace(tensor=SimpleNamespace(shape=(1, 3, 8, 8)))

    metadata = infer_data_input_metadata(config, data)

    assert metadata == DataInputMetadata(kind="image", shape=(1, 3, 8, 8), layout="NCHW")


def test_infer_data_input_metadata_detects_tabular_source(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n")
    config = SimpleNamespace(data=SimpleNamespace(source=str(csv_path)))
    data = SimpleNamespace(tensor=SimpleNamespace(shape=(1, 2)))

    metadata = infer_data_input_metadata(config, data)

    assert metadata == DataInputMetadata(kind="tabular", shape=(1, 2), layout="(B,F)")


def test_source_helpers_detect_directory_contents(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "sample.jpg").write_bytes(b"not-an-image-for-this-test")
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    (table_dir / "features.parquet").write_bytes(b"not-a-parquet-for-this-test")

    assert is_image_source(str(image_dir))
    assert is_tabular_source(str(table_dir))


def test_is_image_source_recurses_into_class_subdirs(tmp_path: Path) -> None:
    # Nested ``ImageFolder`` layout (post-#95): images live under per-class
    # subdirs, not the root. is_image_source must still recognise the dir.
    nested = tmp_path / "test"
    (nested / "NORMAL").mkdir(parents=True)
    (nested / "PNEUMONIA").mkdir(parents=True)
    (nested / "NORMAL" / "IM-0001.jpeg").write_bytes(b"not-an-image-for-this-test")
    (nested / "PNEUMONIA" / "IM-0002.jpeg").write_bytes(b"not-an-image-for-this-test")

    assert is_image_source(str(nested))


def test_is_tabular_source_recurses_into_subdirs(tmp_path: Path) -> None:
    nested = tmp_path / "tabular"
    (nested / "shard_a").mkdir(parents=True)
    (nested / "shard_a" / "rows.csv").write_text("a,b\n1,2\n")

    assert is_tabular_source(str(nested))


def test_is_image_source_matches_uppercase_extensions(tmp_path: Path) -> None:
    # Many real-world dumps use ``.JPG`` / ``.JPEG`` (e.g. camera exports).
    # Detection must be case-insensitive on Linux too — ``Path.rglob`` is
    # case-sensitive there, so the implementation needs an explicit fix.
    image_dir = tmp_path / "uppercase_images"
    image_dir.mkdir()
    (image_dir / "IMG_001.JPG").write_bytes(b"not-an-image-for-this-test")

    assert is_image_source(str(image_dir))


def test_is_tabular_source_matches_uppercase_extensions(tmp_path: Path) -> None:
    table_dir = tmp_path / "uppercase_tables"
    table_dir.mkdir()
    (table_dir / "DATA.CSV").write_text("a,b\n1,2\n")

    assert is_tabular_source(str(table_dir))


def test_shape_tuple_returns_none_for_invalid_shapes() -> None:
    assert shape_tuple(None) is None
    assert shape_tuple(object()) is None
