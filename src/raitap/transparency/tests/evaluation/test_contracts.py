from __future__ import annotations

from raitap.transparency.evaluation.contracts import (
    EvalRequirement,
    EvaluationScore,
    QuantusCategory,
    QuantusMetricSpec,
    SkippedMetric,
)


def test_spec_defaults_and_requires() -> None:
    spec = QuantusMetricSpec(
        category=QuantusCategory.COMPLEXITY,
        quantus_cls="Sparseness",
        requires=frozenset({EvalRequirement.ATTRIBUTIONS}),
        higher_is_better=True,
    )
    assert spec.requires == frozenset({EvalRequirement.ATTRIBUTIONS})
    assert spec.invoker is None
    assert dict(spec.default_kwargs) == {}


def test_score_and_skip_are_plain_records() -> None:
    score = EvaluationScore("sparseness", QuantusCategory.COMPLEXITY, [0.1, 0.2], 0.15, True)
    assert score.aggregate == 0.15
    skip = SkippedMetric("max_sensitivity", frozenset({EvalRequirement.RE_EXPLAIN}), "no explainer")
    assert EvalRequirement.RE_EXPLAIN in skip.missing


def test_evaluation_result_instantiates_and_logs_noop() -> None:
    from raitap.transparency.evaluation.contracts import EvaluationResult

    r = EvaluationResult(
        explanation_name="ig", adapter_target="x", algorithm="IG", scores=[], skipped=[]
    )
    r.log(None)  # no tracker -> no-op, must not raise
    assert r.algorithm == "IG"
