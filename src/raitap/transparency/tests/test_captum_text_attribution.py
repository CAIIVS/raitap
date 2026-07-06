"""Captum text attribution: ``additional_forward_args`` + ``LayerIntegratedGradients`` (#340).

Two layers of coverage:

- ``test_layer_ig_produces_per_token_attributions`` exercises captum directly
  against the tiny HF backend to lock down the mechanism (token ids are
  discrete, so attribution must run over the embedding layer via
  ``LayerIntegratedGradients``, with the attention mask riding as a
  non-attributed ``additional_forward_args`` entry).
- ``test_default_invoker_pops_attention_mask_into_additional_forward_args``
  verifies raitap's own dispatch: ``_default_captum_invoker`` pops a
  designated ``attention_mask`` call kwarg and forwards it as
  ``additional_forward_args`` instead of double-passing it as a normal
  ``attribute()`` kwarg.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import torch

from raitap.transparency.explainers.base_explainer import AttributionInvokeCtx
from raitap.transparency.explainers.captum_explainer import CaptumExplainer, _default_captum_invoker


def test_layer_ig_produces_per_token_attributions() -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("captum")

    from raitap.models.torch_backend import load_hf_text_backend

    backend = load_hf_text_backend(
        "hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
        tokenizer="hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
        device=torch.device("cpu"),
    )
    enc = backend.tokenizer(["hello world"], padding=True, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]

    # Attribution via LayerIntegratedGradients over the embedding layer, mask as
    # additional_forward_args (token ids are discrete -> no gradient on input_ids).
    from captum.attr import LayerIntegratedGradients

    emb = cast("Any", backend.model).get_input_embeddings()
    lig = LayerIntegratedGradients(lambda i, m: backend.model(i, attention_mask=m).logits, emb)
    attr = cast("torch.Tensor", lig.attribute(ids, target=0, additional_forward_args=(mask,)))
    per_token = attr.sum(dim=-1)
    assert per_token.shape == ids.shape


class _RecordingMethod:
    """Fake captum method: records the kwargs ``.attribute()`` is called with."""

    last_call_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, model: object, **init_kwargs: object) -> None:
        self.model = model
        self.init_kwargs = init_kwargs

    def attribute(self, inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        _RecordingMethod.last_call_kwargs = kwargs
        return torch.zeros_like(inputs, dtype=torch.float32)


class _FakeCaptumAttr:
    IntegratedGradients = _RecordingMethod


def test_default_invoker_pops_attention_mask_into_additional_forward_args() -> None:
    explainer = CaptumExplainer(algorithm="IntegratedGradients")
    model = torch.nn.Linear(4, 2)
    inputs = torch.randn(2, 4)
    mask = torch.ones(2, 4)

    ctx = AttributionInvokeCtx(
        explainer=explainer,
        library=_FakeCaptumAttr,
        model=model,
        inputs=inputs,
        input_spec=None,
        call_kwargs={"target": 0, "baselines": None, "attention_mask": mask},
    )

    _default_captum_invoker(ctx)

    recorded = _RecordingMethod.last_call_kwargs
    assert "attention_mask" not in recorded
    additional_forward_args = recorded["additional_forward_args"]
    assert isinstance(additional_forward_args, tuple)
    assert len(additional_forward_args) == 1
    assert additional_forward_args[0] is mask
    assert recorded["target"] == 0


def test_default_invoker_merges_mask_with_caller_additional_forward_args() -> None:
    # A config that already supplies additional_forward_args must keep them AND
    # get the mask appended, not have one clobber the other.
    explainer = CaptumExplainer(algorithm="IntegratedGradients")
    model = torch.nn.Linear(4, 2)
    mask = torch.ones(2, 4)
    caller_arg = object()

    ctx = AttributionInvokeCtx(
        explainer=explainer,
        library=_FakeCaptumAttr,
        model=model,
        inputs=torch.randn(2, 4),
        input_spec=None,
        call_kwargs={
            "target": 0,
            "baselines": None,
            "attention_mask": mask,
            "additional_forward_args": (caller_arg,),
        },
    )

    _default_captum_invoker(ctx)

    recorded = _RecordingMethod.last_call_kwargs
    assert recorded["additional_forward_args"] == (caller_arg, mask)


def test_default_invoker_omits_additional_forward_args_when_no_mask() -> None:
    """Existing image/tabular paths (no ``attention_mask`` kwarg) stay unchanged."""
    explainer = CaptumExplainer(algorithm="IntegratedGradients")
    model = torch.nn.Linear(4, 2)
    inputs = torch.randn(2, 4)

    ctx = AttributionInvokeCtx(
        explainer=explainer,
        library=_FakeCaptumAttr,
        model=model,
        inputs=inputs,
        input_spec=None,
        call_kwargs={"target": 0, "baselines": None},
    )

    _default_captum_invoker(ctx)

    recorded = _RecordingMethod.last_call_kwargs
    assert "additional_forward_args" not in recorded
    assert "attention_mask" not in recorded
