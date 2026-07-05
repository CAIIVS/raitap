"""End-to-end test: real Captum explanation graded by real Quantus (#341)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("torch")

import torch

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.usefixtures("needs_captum", "needs_quantus")
def test_captum_then_quantus_complexity(
    simple_cnn: torch.nn.Module,
    sample_images: torch.Tensor,
    tmp_path: Path,
) -> None:
    from raitap.transparency import CaptumExplainer, QuantusEvaluator
    from raitap.transparency.contracts import InputSpec
    from raitap.transparency.evaluation.semantics import EvaluationContext

    explainer = CaptumExplainer("Saliency")
    explanation = explainer.explain(
        simple_cnn,
        sample_images,
        run_dir=tmp_path / "t",
        target=0,
        raitap_kwargs={
            "input_metadata": InputSpec(
                kind="image",
                shape=tuple(sample_images.shape),
                layout="NCHW",
                metadata={"kind": "image", "layout": "NCHW"},
            )
        },
    )
    ev = QuantusEvaluator(metrics=["sparseness", "complexity"])
    ctx = EvaluationContext(
        result=explanation,
        model=simple_cnn,
        device=torch.device("cpu"),
        explainer=explainer,
        masks=None,
        baseline=None,
        softmax=False,
    )
    out = ev.evaluate(ctx, run_dir=tmp_path)

    ran = {s.metric for s in out.scores}
    assert {"sparseness", "complexity"} <= ran
    assert not out.skipped
    for s in out.scores:
        assert s.aggregate is None or s.aggregate == s.aggregate  # finite or None (not NaN)
