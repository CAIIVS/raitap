from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")

import torch

from raitap.transparency.contracts import ExplanationOutputSpace
from raitap.transparency.evaluation.contracts import EvaluationResult
from raitap.transparency.evaluation.evaluators.quantus_evaluator import QuantusEvaluator
from raitap.transparency.evaluation.semantics import EvaluationContext
from raitap.transparency.tests.evaluation.test_bridge import _make_result

if TYPE_CHECKING:
    from pathlib import Path


class _FakeMetric:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, **call: Any) -> list[float]:
        return [0.5, 0.5, 0.5, 0.5]


def _fake_quantus() -> types.ModuleType:
    mod = types.ModuleType("quantus")
    mod.Sparseness = _FakeMetric  # type: ignore[attr-defined]
    mod.FaithfulnessCorrelation = _FakeMetric  # type: ignore[attr-defined]
    mod.MaxSensitivity = _FakeMetric  # type: ignore[attr-defined]
    return mod


def test_evaluate_runs_static_skips_reexplain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ev = QuantusEvaluator(metrics=["sparseness", "faithfulness_correlation", "max_sensitivity"])
    monkeypatch.setattr(ev, "_lazy_import", lambda submodule=None: _fake_quantus())

    ctx = EvaluationContext(
        result=_make_result(tmp_path, target=1, space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP),
        model=object(),
        device=torch.device("cpu"),
        explainer=None,
        masks=None,
        baseline=None,
        softmax=False,
    )
    out = ev.evaluate(ctx, run_dir=tmp_path)
    assert isinstance(out, EvaluationResult)
    ran = {s.metric for s in out.scores}
    assert "sparseness" in ran and "faithfulness_correlation" in ran
    assert out.scores[0].aggregate == pytest.approx(0.5)
    skipped = {s.metric for s in out.skipped}
    assert "max_sensitivity" in skipped  # no explainer -> RE_EXPLAIN missing
