import pytest

pytest.importorskip("transformers")

import torch

from raitap.models.torch_backend import load_hf_text_backend


def test_hf_text_backend_forwards_ids_and_mask() -> None:
    backend = load_hf_text_backend(
        "hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
        tokenizer="hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
        device=torch.device("cpu"),
    )
    enc = backend.tokenizer(["hello world", "bye"], padding=True, return_tensors="pt")
    out = backend(enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = out.logits if hasattr(out, "logits") else out
    assert logits.shape[0] == 2
