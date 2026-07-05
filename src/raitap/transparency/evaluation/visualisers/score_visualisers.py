"""Bar-chart renderer for quantus explanation-quality scores (#341)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from raitap.transparency.evaluation.contracts import EvaluationResult


class ScoreBarVisualiser:
    """Aggregate score per metric as a labelled bar chart.

    Quantus scores are not attributions, so this does not subclass
    ``BaseVisualiser``: that base's ``validate_explanation`` gate is
    attribution-shaped and does not apply here. This is a standalone
    renderer consuming an ``EvaluationResult`` directly.
    """

    def render(self, evaluation: EvaluationResult) -> Figure:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        labels = [s.metric for s in evaluation.scores if s.aggregate is not None]
        values = [s.aggregate for s in evaluation.scores if s.aggregate is not None]

        fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
        ax.bar(labels, values)
        ax.set_ylabel("aggregate score")
        ax.set_title(f"Explanation quality - {evaluation.algorithm}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig
