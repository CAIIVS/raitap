from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from raitap.transparency.evaluation.contracts import (
    EvaluationResult,
    EvaluationScore,
    QuantusCategory,
)
from raitap.transparency.evaluation.visualisers.score_visualisers import ScoreBarVisualiser


def test_render_returns_figure_with_one_bar_per_metric() -> None:
    ev = EvaluationResult(
        explanation_name="ig",
        adapter_target="x",
        algorithm="IntegratedGradients",
        scores=[
            EvaluationScore("sparseness", QuantusCategory.COMPLEXITY, [0.4], 0.4, True),
            EvaluationScore(
                "faithfulness_correlation", QuantusCategory.FAITHFULNESS, [0.2], 0.2, True
            ),
        ],
    )
    fig = ScoreBarVisualiser().render(ev)
    ax = fig.axes[0]
    assert len(ax.patches) == 2
