"""Cohort visualiser for certified per-logit bound widths from formal verifiers.

For each output class ``k`` over the verified batch, render a boxplot of
``upper[i, k] - lower[i, k]`` widths (NaN-padded rows are dropped). Classes
with no finite samples are omitted from the boxplot but keep their x-axis
tick so the class index remains unambiguous.

Declared compatible with :class:`AssessmentKind.FORMAL_VERIFICATION` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from raitap.robustness.visualisers.registration import robustness_visualiser

from ...contracts import AssessmentKind, ReportFigureScope
from ..base_visualiser import BaseRobustnessVisualiser

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ...contracts import RobustnessVisualisationContext
    from ...results import RobustnessResult


_PLACEHOLDER_NO_BOUNDS = (
    "No output bounds present — configure `compute_output_bounds=True` on MarabouAssessor"
)


def _placeholder_figure(message: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_axis_off()
    return fig


@robustness_visualiser(
    registry_name="output_bounds_cohort",
    supported_assessment_kinds=frozenset({AssessmentKind.FORMAL_VERIFICATION}),
    report_figure_scope=ReportFigureScope.ASSESSOR,
)
class OutputBoundsCohortVisualiser(BaseRobustnessVisualiser):
    """Boxplot of certified per-class bound widths across the verified batch."""

    def __init__(
        self,
        *,
        whis: float | tuple[float, float] = 1.5,
        show_outliers: bool = True,
    ) -> None:
        self.whis = whis
        self.show_outliers = bool(show_outliers)

    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        del kwargs
        bounds = result.output_bounds
        if bounds is None or "lower" not in bounds or "upper" not in bounds:
            return _placeholder_figure(_PLACEHOLDER_NO_BOUNDS)

        lower = np.asarray(bounds["lower"].detach().cpu().numpy(), dtype=float)
        upper = np.asarray(bounds["upper"].detach().cpu().numpy(), dtype=float)
        if lower.size == 0 or upper.size == 0:
            return _placeholder_figure(_PLACEHOLDER_NO_BOUNDS)
        if lower.ndim != 2 or upper.ndim != 2 or lower.shape != upper.shape:
            return _placeholder_figure(_PLACEHOLDER_NO_BOUNDS)

        widths = upper - lower
        finite_mask = np.isfinite(widths)
        if not finite_mask.any():
            return _placeholder_figure(_PLACEHOLDER_NO_BOUNDS)

        n_classes = widths.shape[1]
        per_class: list[np.ndarray] = []
        positions: list[int] = []
        for k in range(n_classes):
            column = widths[:, k]
            finite = column[np.isfinite(column)]
            if finite.size:
                per_class.append(finite)
                positions.append(k)

        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * n_classes), 4.0))
        if per_class:
            ax.boxplot(
                per_class,
                positions=positions,
                whis=self.whis,
                showfliers=self.show_outliers,
            )
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels([f"logit_{k}" for k in range(n_classes)], rotation=0)
        ax.set_xlim(-0.5, n_classes - 0.5)
        ax.set_xlabel("output class")
        ax.set_ylabel("certified width")
        ax.set_title(f"{context.algorithm} — certified bound widths")
        fig.tight_layout()
        return fig
