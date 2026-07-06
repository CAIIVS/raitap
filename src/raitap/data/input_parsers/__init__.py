"""Input-parser family package. Importing fires @input_parser decorators."""

from __future__ import annotations

from .text_csv import TextCsvInputParser  # pyright: ignore[reportUnusedImport]
from .text_dir import TextDirInputParser  # pyright: ignore[reportUnusedImport]
from .text_jsonl import TextJsonlInputParser  # pyright: ignore[reportUnusedImport]

__all__ = ["TextCsvInputParser", "TextDirInputParser", "TextJsonlInputParser"]
