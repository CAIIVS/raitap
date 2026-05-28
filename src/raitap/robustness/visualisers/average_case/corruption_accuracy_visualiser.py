"""Average-case (statistical-sampling) accuracy visualiser.

Two bars — clean vs corrupted accuracy — with a CI whisker on the corrupted bar,
annotated with corruption name, severity, and N.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from raitap.robustness.visualisers.registration import robustness_visualiser

from ...contracts import AssessmentKind, PerturbationDistribution
from ..base_visualiser import BaseRobustnessVisualiser

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ...contracts import RobustnessVisualisationContext
    from ...results import RobustnessResult


@robustness_visualiser(
    registry_name="corruption_accuracy",
    supported_assessment_kinds=frozenset({AssessmentKind.STATISTICAL_SAMPLING}),
)
class CorruptionAccuracyVisualiser(BaseRobustnessVisualiser):
    """Clean vs corrupted accuracy bars with a CI whisker."""

    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        del kwargs
        metrics = result.metrics
        clean = float(metrics.clean_accuracy)
        corrupted = float(metrics.corrupted_accuracy or 0.0)
        ci_low = float(
            metrics.accuracy_ci_low if metrics.accuracy_ci_low is not None else corrupted
        )
        ci_high = float(
            metrics.accuracy_ci_high if metrics.accuracy_ci_high is not None else corrupted
        )

        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ["clean", "corrupted"]
        values = [clean, corrupted]
        lower = [0.0, max(0.0, corrupted - ci_low)]
        upper = [0.0, max(0.0, ci_high - corrupted)]
        ax.bar(labels, values, color=["#1f77b4", "#d62728"])
        ax.errorbar(labels, values, yerr=[lower, upper], fmt="none", ecolor="black", capsize=5)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("accuracy")

        region = result.semantics.perturbation
        if isinstance(region, PerturbationDistribution):
            subtitle = (
                f"{region.corruption_name} (severity {region.severity}), N={metrics.n_samples}"
            )
        else:
            subtitle = f"N={metrics.n_samples}"
        ax.set_title(f"{context.algorithm} — average-case accuracy\n{subtitle}", fontsize=11)

        for i, value in enumerate(values):
            ax.text(i, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        return fig
