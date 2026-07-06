import torch

from raitap.data.metadata import infer_data_input_metadata
from raitap.data.types import InputModality


class _Data:
    input_metadata = None
    tensor = torch.zeros((2, 8), dtype=torch.long)
    input_modality = InputModality.text
    source = ""


def test_text_modality_infers_text_kind_and_tokens_layout() -> None:
    md = infer_data_input_metadata(config=object(), data=_Data())
    assert md.kind == "text"
    assert md.layout == "TOKENS"
