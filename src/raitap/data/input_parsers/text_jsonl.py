"""JSONL text input parser (one object per line, reads a text field)."""

from __future__ import annotations

import json
from pathlib import Path

from raitap.configs.schema import TextJsonlInputsConfig
from raitap.data.data import SourceKind, get_source_path
from raitap.data.input_parsers.registration import input_parser
from raitap.data.types import InputModality


@input_parser(registry_name="text_jsonl", schema=TextJsonlInputsConfig)
class TextJsonlInputParser:
    supported_modalities = frozenset({InputModality.text})

    def __init__(self, *, text_field: str = "text") -> None:
        self.text_field = text_field

    def parse(self, *, source: str) -> list[str]:
        path = get_source_path(source, kind=SourceKind.DATA)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return [str(json.loads(line)[self.text_field]) for line in lines if line.strip()]
