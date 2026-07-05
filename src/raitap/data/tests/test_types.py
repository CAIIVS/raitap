from raitap.data.types import MODALITY_EXTENSIONS, InputModality


def test_input_modality_has_only_todays_members() -> None:
    assert {m.value for m in InputModality} == {"image", "tabular", "text"}


def test_modality_extensions_cover_every_member() -> None:
    assert set(MODALITY_EXTENSIONS) == set(InputModality)


def test_modality_extensions_values() -> None:
    assert ".png" in MODALITY_EXTENSIONS[InputModality.image]
    assert ".jpeg" in MODALITY_EXTENSIONS[InputModality.image]
    assert ".csv" in MODALITY_EXTENSIONS[InputModality.tabular]
    assert ".parquet" in MODALITY_EXTENSIONS[InputModality.tabular]
