"""Instantiation factory for label parsers (delegates to _parser_factory)."""

from __future__ import annotations

from typing import Any

from raitap.data._parser_factory import create_parser
from raitap.data.label_parsers.base import LabelParser  # noqa: TC001  runtime-resolvable


def create_label_parser(labels_config: Any) -> LabelParser:
    return create_parser(labels_config, prefix="raitap.data.label_parsers.", kind="label parser")
