import dataclasses

import pytest

from raitap.configs.schema import CocoLabelsConfig, DirectoryLabelsConfig


def test_coco_config_has_no_tabular_fields() -> None:
    names = {f.name for f in dataclasses.fields(CocoLabelsConfig)}
    assert "id_column" not in names
    assert "column" not in names
    assert "encoding" not in names
    assert {"_target_", "source", "id_strategy"} <= names


def test_directory_config_has_only_target() -> None:
    names = {f.name for f in dataclasses.fields(DirectoryLabelsConfig)}
    assert names == {"_target_"}


def test_labelformat_enum_is_gone() -> None:
    with pytest.raises(ImportError):
        from raitap.data.types import LabelFormat  # noqa: F401
