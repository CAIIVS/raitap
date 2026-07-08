from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")

import torch

import raitap.transparency.evaluation.step as step_mod
from raitap.configs.registry_resolve import UnsafeConfigTargetError
from raitap.configs.schema import EvaluationConfig
from raitap.transparency.contracts import ExplanationOutputSpace
from raitap.transparency.tests.evaluation.test_bridge import _make_result

if TYPE_CHECKING:
    from pathlib import Path


class _StubBackend:
    device = torch.device("cpu")

    def autograd_module(self) -> torch.nn.Module:
        return torch.nn.Identity()


class _StubPrepared:
    backend = _StubBackend()
    explainer = None


def test_grade_returns_empty_when_no_config(tmp_path: Path) -> None:
    result = _make_result(tmp_path, target=0, space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP)
    assert step_mod.grade_explanations(None, [result], _StubPrepared()) == []


def test_grade_returns_empty_when_no_explanations(tmp_path: Path) -> None:
    cfg = EvaluationConfig(use="quantus", metrics=["sparseness"])
    assert step_mod.grade_explanations(cfg, [], _StubPrepared()) == []


def test_grade_resolves_use_key_to_quantus_evaluator_fqn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``use: quantus`` config resolves through the registry to ``QuantusEvaluator``,
    not to an arbitrary dotted path (#301)."""
    captured: dict[str, Any] = {}

    class _FakeEval:
        def evaluate(self, ctx: Any, *, run_dir: Any) -> Any:
            from raitap.transparency.evaluation.contracts import EvaluationResult

            return EvaluationResult(
                explanation_name=ctx.result.name,
                adapter_target="x",
                algorithm=ctx.result.algorithm,
                scores=[],
                skipped=[],
            )

    def _fake_instantiate(cfg: dict[str, Any]) -> _FakeEval:
        captured.update(cfg)
        return _FakeEval()

    monkeypatch.setattr(step_mod, "instantiate", _fake_instantiate)
    cfg = EvaluationConfig(use="quantus", metrics=["sparseness"])
    result = _make_result(tmp_path, target=0, space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP)

    step_mod.grade_explanations(cfg, [result], _StubPrepared())

    assert "use" not in captured
    assert captured["_target_"].endswith(".QuantusEvaluator")


def test_grade_rejects_dotted_target_config(tmp_path: Path) -> None:
    """A shipped ``_target_: raitap.transparency.QuantusEvaluator`` config must be
    rejected, not instantiated (closes the dotted-path RCE surface, #301)."""
    cfg = {"_target_": "raitap.transparency.QuantusEvaluator", "metrics": ["sparseness"]}
    result = _make_result(tmp_path, target=0, space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP)

    with pytest.raises(UnsafeConfigTargetError):
        step_mod.grade_explanations(cfg, [result], _StubPrepared())


def test_grade_instantiates_and_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEval:
        def evaluate(self, ctx: Any, *, run_dir: Any) -> Any:
            from raitap.transparency.evaluation.contracts import EvaluationResult

            return EvaluationResult(
                explanation_name=ctx.result.name,
                adapter_target="x",
                algorithm=ctx.result.algorithm,
                scores=[],
                skipped=[],
            )

    monkeypatch.setattr(step_mod, "instantiate", lambda cfg: _FakeEval())
    cfg = EvaluationConfig(use="quantus", metrics=["sparseness"])
    result = _make_result(tmp_path, target=0, space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP)
    out = step_mod.grade_explanations(cfg, [result], _StubPrepared())
    assert len(out) == 1


def test_grade_writes_evaluations_json_to_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``grade_explanations`` persists each result to ``<run_dir>/evaluations.json`` (#341)."""
    import json

    class _FakeEval:
        def evaluate(self, ctx: Any, *, run_dir: Any) -> Any:
            from raitap.transparency.evaluation.contracts import (
                EvaluationResult,
                EvaluationScore,
                QuantusCategory,
            )

            return EvaluationResult(
                explanation_name=ctx.result.name,
                adapter_target="x",
                algorithm=ctx.result.algorithm,
                scores=[
                    EvaluationScore("sparseness", QuantusCategory.COMPLEXITY, [0.4], 0.4, True)
                ],
                skipped=[],
                run_dir=run_dir,
            )

    monkeypatch.setattr(step_mod, "instantiate", lambda cfg: _FakeEval())
    cfg = EvaluationConfig(use="quantus", metrics=["sparseness"])
    result = _make_result(tmp_path, target=0, space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP)

    step_mod.grade_explanations(cfg, [result], _StubPrepared())

    written = tmp_path / "evaluations.json"
    assert written.exists()
    payload = json.loads(written.read_text())
    assert payload["algorithm"] == "IntegratedGradients"
    assert payload["scores"][0]["metric"] == "sparseness"


def test_grade_puts_model_in_eval_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MODEL-requiring Quantus metrics raise ``AttributeError`` unless the raw
    model pulled off the backend is switched to eval mode before grading (#341)."""
    module = torch.nn.Identity()
    module.train()  # start in train mode; grade_explanations must flip it

    class _EvalModeStubBackend:
        device = torch.device("cpu")

        def autograd_module(self) -> torch.nn.Module:
            return module

    class _EvalModeStubPrepared:
        backend = _EvalModeStubBackend()
        explainer = None

    class _FakeEval:
        def evaluate(self, ctx: Any, *, run_dir: Any) -> Any:
            from raitap.transparency.evaluation.contracts import EvaluationResult

            return EvaluationResult(
                explanation_name=ctx.result.name,
                adapter_target="x",
                algorithm=ctx.result.algorithm,
                scores=[],
                skipped=[],
            )

    monkeypatch.setattr(step_mod, "instantiate", lambda cfg: _FakeEval())
    cfg = EvaluationConfig(use="quantus", metrics=["sparseness"])
    result = _make_result(tmp_path, target=0, space=ExplanationOutputSpace.IMAGE_SPATIAL_MAP)

    assert module.training is True
    step_mod.grade_explanations(cfg, [result], _EvalModeStubPrepared())
    assert module.training is False
