from __future__ import annotations

from dataclasses import fields

from raitap.configs.schema import EvaluationConfig, TransparencyConfig


def test_transparency_config_has_optional_evaluation() -> None:
    names = {f.name for f in fields(TransparencyConfig)}
    assert "evaluation" in names


def test_evaluation_config_shape() -> None:
    cfg = EvaluationConfig(_target_="raitap.transparency.QuantusEvaluator", metrics=["sparseness"])
    assert cfg.metrics == ["sparseness"]
    assert cfg.constructor == {}
