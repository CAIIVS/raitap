"""Instantiation factory for input parsers (delegates to _parser_factory)."""

from __future__ import annotations

from typing import Any

from raitap.data._parser_factory import create_parser
from raitap.data.input_parsers.base import InputParser  # noqa: TC001  runtime-resolvable


def create_input_parser(inputs_config: Any) -> InputParser:
    return create_parser(inputs_config, group="data/inputs", kind="input parser")
