from __future__ import annotations

import json
from typing import TYPE_CHECKING

from raitap.transparency.evaluation.contracts import (
    EvalRequirement,
    EvaluationResult,
    EvaluationScore,
    QuantusCategory,
    QuantusMetricSpec,
    SkippedMetric,
)

if TYPE_CHECKING:
    from pathlib import Path


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
    r = EvaluationResult(
        explanation_name="ig", adapter_target="x", algorithm="IG", scores=[], skipped=[]
    )
    r.log(None)  # no tracker -> no-op, must not raise
    assert r.algorithm == "IG"


def test_write_artifacts_noop_when_no_run_dir() -> None:
    r = EvaluationResult(
        explanation_name="ig", adapter_target="x", algorithm="IG", scores=[], skipped=[]
    )
    r.write_artifacts()  # run_dir is None -> must not raise, writes nothing


def test_write_artifacts_writes_evaluations_json(tmp_path: Path) -> None:
    r = EvaluationResult(
        explanation_name="ig",
        adapter_target="raitap.transparency.CaptumExplainer",
        algorithm="IntegratedGradients",
        scores=[
            EvaluationScore("sparseness", QuantusCategory.COMPLEXITY, [0.1, 0.2], 0.15, True),
            EvaluationScore(
                "model_parameter_randomisation",
                QuantusCategory.RANDOMISATION,
                [0.5],
                0.5,
                None,
            ),
        ],
        skipped=[
            SkippedMetric(
                "pointing_game",
                frozenset({EvalRequirement.SEGMENTATION}),
                "no segmentation mask available",
            ),
        ],
        run_dir=tmp_path,
    )
    r.write_artifacts()

    written = tmp_path / "evaluations.json"
    assert written.exists()
    payload = json.loads(written.read_text())

    assert payload["explanation_name"] == "ig"
    assert payload["algorithm"] == "IntegratedGradients"
    assert payload["adapter_target"] == "raitap.transparency.CaptumExplainer"
    assert payload["scores"] == [
        {
            "metric": "sparseness",
            "category": "complexity",
            "aggregate": 0.15,
            "values": [0.1, 0.2],
            "higher_is_better": True,
        },
        {
            "metric": "model_parameter_randomisation",
            "category": "randomisation",
            "aggregate": 0.5,
            "values": [0.5],
            "higher_is_better": None,
        },
    ]
    assert payload["skipped"] == [
        {
            "metric": "pointing_game",
            "missing": ["segmentation"],
            "message": "no segmentation mask available",
        },
    ]
