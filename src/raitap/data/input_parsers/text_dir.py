"""Directory-of-.txt text input parser (one document per file, sorted by name)."""

from __future__ import annotations

from pathlib import Path

from raitap.configs.schema import TextDirInputsConfig
from raitap.data.data import SourceKind, get_source_path
from raitap.data.input_parsers.registration import input_parser
from raitap.data.types import InputModality


@input_parser(registry_name="text_dir", schema=TextDirInputsConfig)
class TextDirInputParser:
    supported_modalities = frozenset({InputModality.text})

    def __init__(self, *, source: str) -> None:
        self.source = source

    def parse(self, *, source: str, sample_ids: list[str] | None) -> list[str]:
        root = Path(get_source_path(self.source, kind=SourceKind.DATA))
        files = sorted(root.glob("*.txt"))
        return [f.read_text(encoding="utf-8") for f in files]
