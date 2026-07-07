from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import torch

from raitap.data.metadata import infer_data_input_metadata
from raitap.data.types import InputModality

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig


class _Data:
    input_metadata = None
    tensor = torch.zeros((2, 8), dtype=torch.long)
    input_modality = InputModality.text
    source = ""


def test_text_modality_infers_text_kind_and_tokens_layout() -> None:
    config = cast("AppConfig", SimpleNamespace(data=SimpleNamespace(input_metadata=None)))
    md = infer_data_input_metadata(config=config, data=_Data())
    assert md.kind == "text"
    assert md.layout == "TOKENS"
