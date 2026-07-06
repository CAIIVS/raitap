"""Directory-of-documents text input parser (one document per file, sorted by name).

Reads every file whose suffix is a recognised text extension
(``MODALITY_EXTENSIONS[InputModality.text]``, currently ``.txt`` and ``.md``).
"""

from __future__ import annotations

from pathlib import Path

from raitap.configs.schema import TextDirInputsConfig
from raitap.data.data import SourceKind, get_source_path
from raitap.data.input_parsers.registration import input_parser
from raitap.data.types import MODALITY_EXTENSIONS, InputModality


@input_parser(registry_name="text_dir", schema=TextDirInputsConfig)
class TextDirInputParser:
    supported_modalities = frozenset({InputModality.text})

    def parse(self, *, source: str) -> list[str]:
        exts = MODALITY_EXTENSIONS[InputModality.text]
        root = Path(get_source_path(source, kind=SourceKind.DATA))
        files = sorted(f for f in root.iterdir() if f.suffix.lower() in exts)
        return [f.read_text(encoding="utf-8") for f in files]
