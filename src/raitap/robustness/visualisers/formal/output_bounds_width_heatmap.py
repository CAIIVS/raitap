"""Width heatmap visualiser for certified per-logit output bounds.

For each sample row ``i`` and output class ``k``, render
``width[i, k] = upper[i, k] - lower[i, k]`` as a heatmap cell. Rows where the
verifier did not certify any bound (NaN) are rendered as masked / grey cells
so the visual is honest about coverage.

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


def _auto_figsize(n: int, k: int) -> tuple[float, float]:
    return (max(k * 0.6 + 2, 6), max(n * 0.25 + 1.5, 3))


@robustness_visualiser(
    registry_name="output_bounds_width_heatmap",
    supported_assessment_kinds={AssessmentKind.FORMAL_VERIFICATION},
    report_figure_scope=ReportFigureScope.ASSESSOR,
)
class OutputBoundsWidthHeatmapVisualiser(BaseRobustnessVisualiser):
    """Heatmap of certified per-class output-bound widths across the batch."""

    def __init__(
        self,
        *,
        cmap: str = "viridis",
        max_samples: int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        self.cmap = cmap
        self.max_samples = max(int(max_samples), 1) if max_samples is not None else None
        self.figsize = figsize

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
        if lower.size == 0 or upper.size == 0 or lower.shape != upper.shape or lower.ndim != 2:
            return _placeholder_figure(_PLACEHOLDER_NO_BOUNDS)

        widths = upper - lower
        if not np.isfinite(widths).any():
            return _placeholder_figure(_PLACEHOLDER_NO_BOUNDS)

        n, k = widths.shape
        if self.max_samples is not None and n > self.max_samples:
            widths = widths[: self.max_samples]
            n = self.max_samples

        masked = np.ma.masked_invalid(widths)
        figsize = self.figsize or _auto_figsize(n, k)
        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap(self.cmap).copy()
        cmap.set_bad(color="#bfbfbf")
        im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")

        ax.set_xticks(range(k))
        ax.set_xticklabels([f"logit_{j}" for j in range(k)])
        ax.set_yticks(range(n))
        sample_names = context.sample_names or []
        if context.show_sample_names and sample_names:
            ax.set_yticklabels(
                [sample_names[i] if i < len(sample_names) else str(i) for i in range(n)]
            )
        else:
            ax.set_yticklabels([str(i) for i in range(n)])
        ax.set_xlabel("output class")
        ax.set_ylabel("sample")
        ax.set_title(f"{context.algorithm} — certified bound widths")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("certified bound width")
        fig.tight_layout()
        return fig
