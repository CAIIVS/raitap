"""CSV/TSV/Parquet text input parser (reads one text column)."""

from __future__ import annotations

from raitap.configs.schema import TextCsvInputsConfig
from raitap.data.data import SourceKind, _load_tabular_frame, get_source_path
from raitap.data.input_parsers.registration import input_parser
from raitap.data.types import InputModality


@input_parser(registry_name="text_csv", schema=TextCsvInputsConfig)
class TextCsvInputParser:
    supported_modalities = frozenset({InputModality.text})

    def __init__(self, *, source: str, text_column: str, id_column: str | None = None) -> None:
        self.source = source
        self.text_column = text_column
        self.id_column = id_column

    def parse(self, *, source: str, sample_ids: list[str] | None) -> list[str]:
        path = get_source_path(self.source, kind=SourceKind.DATA)
        frame = _load_tabular_frame(path)
        return [str(v) for v in frame[self.text_column].tolist()]
