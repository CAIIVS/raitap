import pytest

pytest.importorskip("transformers")

import torch
from transformers import AutoTokenizer

from raitap.data.data import _tokenise_texts


def test_tokenise_texts_returns_ids_and_mask() -> None:
    tok = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    )
    ids, mask = _tokenise_texts(["hello world", "bye"], tokenizer=tok, max_length=16)
    assert ids.dtype == torch.long and ids.shape == mask.shape
    assert ids.shape[0] == 2
    assert mask.sum() > 0
