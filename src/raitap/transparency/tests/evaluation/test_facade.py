from __future__ import annotations


def test_evaluator_symbol_exposed_on_transparency() -> None:
    import raitap.transparency as t

    assert hasattr(t, "QuantusEvaluator")
    assert hasattr(t, "ScoreBarVisualiser")
