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

from raitap.robustness.visualisers.registration import robustness_visualiser

from ...contracts import AssessmentKind
from ..base_visualiser import BaseRobustnessVisualiser

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from ...contracts import RobustnessVisualisationContext
    from ...results import RobustnessResult


_PLACEHOLDER_NO_BOUNDS = (
    "No output bounds present — configure `compute_output_bounds=True` on MarabouAssessor"
)
_NO_BOUND_COLOR = "#bfbfbf"


def _placeholder_figure(message: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_axis_off()
    return fig


@robustness_visualiser(
    registry_name="output_bounds_pinned",
    supported_assessment_kinds=frozenset({AssessmentKind.FORMAL_VERIFICATION}),
)
class OutputBoundsPinnedVisualiser(BaseRobustnessVisualiser):
    """Per-pinned-sample plot of ``[lower_k, upper_k]`` certified intervals."""

    def __init__(
        self,
        *,
        max_samples: int = 4,
        target_color: str = "#d62728",
        bar_color: str = "#1f77b4",
        sample_indices: Sequence[int] | None = None,
    ) -> None:
        self.max_samples = max(int(max_samples), 1)
        self.target_color = target_color
        self.bar_color = bar_color
        self.sample_indices: tuple[int, ...] | None = (
            tuple(int(i) for i in sample_indices) if sample_indices is not None else None
        )

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
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), squeeze=False)

        for row_idx, sample_index in enumerate(selected):
            ax = axes[row_idx][0]
            target_k = int(targets[sample_index]) if sample_index < len(targets) else -1
            for k in range(n_classes):
                lo = lower[sample_index, k]
                hi = upper[sample_index, k]
                color = self.target_color if k == target_k else self.bar_color
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    ax.hlines(k, -0.5, 0.5, colors=_NO_BOUND_COLOR, linewidth=1.0)
                    continue
                ax.hlines(k, lo, hi, colors=color, linewidth=3.0)
                ax.plot([lo, hi], [k, k], "|", color=color, markersize=8)

            ax.set_yticks(range(n_classes))
            ax.set_yticklabels([f"logit_{k}" for k in range(n_classes)])
            ax.invert_yaxis()
            ax.set_xlabel("certified value")

            title = f"sample {sample_index} | target={target_k}"
            if context.show_sample_names and sample_names and sample_index < len(sample_names):
                title = f"{sample_names[sample_index]} ({title})"
            ax.set_title(title)

        fig.suptitle(f"{context.algorithm} — pinned output bounds", fontsize=12)
        fig.tight_layout()
        return fig
