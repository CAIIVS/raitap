"""Pinned-sample visualiser for certified per-logit output bounds.

For each selected sample row ``i``, draw a horizontal interval per output
class ``k`` covering ``[lower[i, k], upper[i, k]]``. Highlight the target
class's interval. Classes with NaN bounds get a thin grey "no bound" marker.

Sample selection priority:

1. Explicit ``sample_indices`` kwarg passed to the visualiser constructor.
2. Otherwise, the first ``max_samples`` rows where neither ``lower[i]`` nor
   ``upper[i]`` is all-NaN.

Note: the issue spec mentions reading resolved pin indices off
``result.semantics.sample_selection`` — at the time of writing the public
``SampleSelection`` contract is batch-wide IDs/names without per-pin indices,
so the constructor-kwarg path is the only pinning surface this visualiser
exposes.

Declared compatible with :class:`AssessmentKind.FORMAL_VERIFICATION` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from raitap.robustness.visualisers.registration import robustness_visualiser

from ...contracts import AssessmentKind, ReportFigureScope
from ..base_visualiser import BaseRobustnessVisualiser

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from ...contracts import RobustnessVisualisationContext
    from ...results import RobustnessResult


_PLACEHOLDER_NO_BOUNDS = (
    "No output bounds present — use a verifier that emits certified bounds "
    "(AutoLiRPAAssessor, or MarabouAssessor with `compute_output_bounds=True`)"
)
_NO_BOUND_COLOR = "#bfbfbf"


def _placeholder_figure(message: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_axis_off()
    return fig


@robustness_visualiser(
    registry_name="output_bounds_pinned",
    supported_assessment_kinds={AssessmentKind.FORMAL_VERIFICATION},
    report_figure_scope=ReportFigureScope.ASSESSOR,
)
class OutputBoundsPinnedVisualiser(BaseRobustnessVisualiser):
    """Per-pinned-sample plot of ``[lower_k, upper_k]`` certified intervals."""

    def __init__(
        self,
        *,
        max_samples: int = 4,
        max_classes: int = 20,
        target_color: str = "#d62728",
        bar_color: str = "#1f77b4",
        sample_indices: Sequence[int] | None = None,
    ) -> None:
        self.max_samples = max(int(max_samples), 1)
        # Cap the number of output classes drawn per subplot. Above this, one
        # row + y-tick per class (e.g. 1000 for ImageNet) collapses into an
        # unreadable smear, so we show the target plus the classes with the
        # largest certified upper bounds (the competitors that decide whether
        # the target is dominated). At or below the cap, every class is shown
        # exactly as before.
        self.max_classes = max(int(max_classes), 1)
        self.target_color = target_color
        self.bar_color = bar_color
        self.sample_indices: tuple[int, ...] | None = (
            tuple(int(i) for i in sample_indices) if sample_indices is not None else None
        )

    def _select_classes(self, upper_row: np.ndarray, target_k: int, n_classes: int) -> list[int]:
        """Return the class indices to draw for one sample (ascending order)."""
        if n_classes <= self.max_classes:
            return list(range(n_classes))
        # Rank by certified upper bound (NaN/no-bound classes sink to the bottom).
        ranked = np.argsort(np.nan_to_num(upper_row, nan=-np.inf))[::-1]
        chosen: list[int] = [int(k) for k in ranked[: self.max_classes]]
        if 0 <= target_k < n_classes and target_k not in chosen:
            chosen[-1] = target_k  # guarantee the target interval is visible.
        return sorted(set(chosen))

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

        n_rows, n_classes = lower.shape
        finite_rows = [
            i for i in range(n_rows) if np.isfinite(lower[i]).any() and np.isfinite(upper[i]).any()
        ]

        if self.sample_indices is not None:
            selected = [i for i in self.sample_indices if 0 <= i < n_rows]
        else:
            selected = finite_rows[: self.max_samples]

        if not selected:
            return _placeholder_figure(_PLACEHOLDER_NO_BOUNDS)

        targets = result.targets.detach().cpu().numpy()
        sample_names = context.sample_names or []

        n = len(selected)
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), squeeze=False, layout="constrained")

        drew_no_bound = False
        for row_idx, sample_index in enumerate(selected):
            ax = axes[row_idx][0]
            target_k = int(targets[sample_index]) if sample_index < len(targets) else -1
            shown = self._select_classes(upper[sample_index], target_k, n_classes)
            for pos, k in enumerate(shown):
                lo = lower[sample_index, k]
                hi = upper[sample_index, k]
                color = self.target_color if k == target_k else self.bar_color
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    ax.hlines(pos, -0.5, 0.5, colors=_NO_BOUND_COLOR, linewidth=1.0)
                    drew_no_bound = True
                    continue
                ax.hlines(pos, lo, hi, colors=color, linewidth=3.0)
                ax.plot([lo, hi], [pos, pos], "|", color=color, markersize=8)

            ax.set_yticks(range(len(shown)))
            ax.set_yticklabels([f"logit_{k}" for k in shown])
            ax.invert_yaxis()
            ax.set_xlabel("Certified value")

            title = f"sample {sample_index} | target={target_k}"
            if len(shown) < n_classes:
                title += f" | top {len(shown)}/{n_classes} classes"
            if context.show_sample_names and sample_names and sample_index < len(sample_names):
                title = f"{sample_names[sample_index]} ({title})"
            ax.set_title(title)

        # Figure-level legend (placed OUTSIDE the axes so it never sits on top of
        # the bars) so the red/blue/grey coding is readable without the docs: each
        # bar is a class's certified [lower, upper] logit range; VERIFIED means the
        # target bar sits fully right of every other bar.
        handles = [
            Line2D([0], [0], color=self.target_color, lw=3, label="target class"),
            Line2D([0], [0], color=self.bar_color, lw=3, label="other classes"),
        ]
        if drew_no_bound:
            handles.append(
                Line2D([0], [0], color=_NO_BOUND_COLOR, lw=1, label="no certified bound")
            )
        fig.legend(handles=handles, loc="outside upper right", fontsize=8, framealpha=0.9)

        fig.suptitle(f"{context.algorithm} — pinned output bounds", fontsize=12)
        return fig
