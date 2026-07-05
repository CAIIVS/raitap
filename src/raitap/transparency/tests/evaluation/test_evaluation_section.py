from __future__ import annotations

from typing import TYPE_CHECKING

from raitap.transparency.evaluation.contracts import (
    EvalRequirement,
    EvaluationResult,
    EvaluationScore,
    QuantusCategory,
    SkippedMetric,
)
from raitap.transparency.report import TransparencyPhaseResult, _build_evaluation_section

if TYPE_CHECKING:
    from pathlib import Path


def _make_evaluation() -> EvaluationResult:
    return EvaluationResult(
        explanation_name="ig",
        adapter_target="raitap.transparency.CaptumExplainer",
        algorithm="IntegratedGradients",
        scores=[
            EvaluationScore("sparseness", QuantusCategory.COMPLEXITY, [0.1, 0.2], 0.15, True),
            EvaluationScore("pixel_flipping", QuantusCategory.FAITHFULNESS, [0.3], 0.3, False),
            EvaluationScore(
                "model_parameter_randomisation",
                QuantusCategory.RANDOMISATION,
                [0.5],
                0.5,
                None,
            ),
            EvaluationScore(
                "broken_metric",
                QuantusCategory.AXIOMATIC,
                [],
                None,
                True,
            ),
        ],
        skipped=[
            SkippedMetric(
                "pointing_game",
                frozenset({EvalRequirement.SEGMENTATION}),
                "no segmentation mask available",
            ),
        ],
    )


def test_build_evaluation_section_returns_none_when_no_evaluations(tmp_path: Path) -> None:
    outputs = TransparencyPhaseResult(explanations=[], evaluations=[])
    assert _build_evaluation_section(outputs, assets_dir=tmp_path) is None


def test_build_evaluation_section_table_rows(tmp_path: Path) -> None:
    evaluation = _make_evaluation()
    outputs = TransparencyPhaseResult(explanations=[], evaluations=[evaluation])

    section = _build_evaluation_section(outputs, assets_dir=tmp_path)

    assert section is not None
    assert section.title == "Explanation quality (Quantus)"
    assert len(section.groups) == 1
    group = section.groups[0]
    assert group.heading == "ig"
    assert group.table_rows == (
        ("sparseness", "0.1500 ↑"),
        ("pixel_flipping", "0.3000 ↓"),
        ("model_parameter_randomisation", "0.5000"),
        ("broken_metric", "n/a"),
        ("pointing_game", "skipped: no segmentation mask available"),
    )


def test_build_evaluation_section_uses_algorithm_when_no_explanation_name(
    tmp_path: Path,
) -> None:
    evaluation = EvaluationResult(
        explanation_name=None,
        adapter_target="x",
        algorithm="Saliency",
        scores=[],
        skipped=[],
    )
    outputs = TransparencyPhaseResult(explanations=[], evaluations=[evaluation])

    section = _build_evaluation_section(outputs, assets_dir=tmp_path)

    assert section is not None
    assert section.groups[0].heading == "Saliency"
