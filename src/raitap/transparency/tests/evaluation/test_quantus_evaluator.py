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


class _FakeDictMetric:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, **call: Any) -> dict[str, list[float]]:
        return {"layer.0": [0.1, 0.2], "layer.1": [0.3, 0.4]}


def _fake_quantus_dict() -> types.ModuleType:
    mod = types.ModuleType("quantus")
    mod.FaithfulnessCorrelation = _FakeDictMetric  # type: ignore[attr-defined]
    return mod


def test_evaluate_flattens_dict_valued_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Quantus 0.6.0's ModelParameterRandomisation (and others) can return
    # dict[str, list[float]] keyed by layer name instead of a flat list; reuse
    # ``faithfulness_correlation`` here since it only needs ATTRIBUTIONS + MODEL
    # (no explainer required to satisfy RE_EXPLAIN).
    ev = QuantusEvaluator(metrics=["faithfulness_correlation"])
    monkeypatch.setattr(ev, "_lazy_import", lambda submodule=None: _fake_quantus_dict())

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
    assert not out.skipped
    score = next(s for s in out.scores if s.metric == "faithfulness_correlation")
    assert score.values == [0.1, 0.2, 0.3, 0.4]


class _FakeRaisingMetric:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, **call: Any) -> list[float]:
        raise RuntimeError("boom")


def _fake_quantus_raising() -> types.ModuleType:
    mod = types.ModuleType("quantus")
    mod.Sparseness = _FakeRaisingMetric  # type: ignore[attr-defined]
    return mod


def test_evaluate_skips_metric_that_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ev = QuantusEvaluator(metrics=["sparseness"])
    monkeypatch.setattr(ev, "_lazy_import", lambda submodule=None: _fake_quantus_raising())

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
    assert out.scores == []
    assert len(out.skipped) == 1
    assert out.skipped[0].metric == "sparseness"
    assert "boom" in out.skipped[0].message


def test_instantiates_from_evaluation_config() -> None:
    from hydra.utils import instantiate

    from raitap.configs.registry_resolve import resolve_target_fqn
    from raitap.configs.schema import EvaluationConfig
    from raitap.configs.utils import cfg_to_dict

    cfg = cfg_to_dict(
        EvaluationConfig(use="quantus", metrics=["sparseness"], raitap={"softmax": True})
    )
    cfg["_target_"] = resolve_target_fqn("_unscoped", cfg.pop("use"))
    ev = instantiate(cfg)
    assert type(ev).__name__ == "QuantusEvaluator"
    assert ev.metrics == ["sparseness"]
    assert ev.softmax is True
