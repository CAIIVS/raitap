"""Base protocol for input-source parsers (bytes/column -> list[str])."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from raitap.data.types import InputModality  # noqa: TC001  runtime-resolvable for get_type_hints()


@runtime_checkable
class InputParser(Protocol):
    """Protocol every input-source parser must satisfy."""

    supported_modalities: frozenset[InputModality]

    def parse(self, *, source: str, sample_ids: list[str] | None) -> list[str]: ...
