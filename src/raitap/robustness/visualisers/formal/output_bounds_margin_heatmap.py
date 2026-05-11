"""Margin heatmap visualiser for certified per-logit output bounds.

For each sample row ``i`` and output class ``k``, render
``margin[i, k] = lower[i, target_i] - upper[i, k]`` as a heatmap cell.

* Positive margin → the target class is *provably* above class ``k`` for every
  input in the certified region.
* Negative margin → class ``k`` could overtake the target somewhere in the
  certified region; the verifier has not ruled out a tie / flip at that
  logit.

The diverging colormap is centred on zero via
:class:`matplotlib.colors.TwoSlopeNorm`, so colours stay interpretable when
the magnitudes of the positive and negative tails differ.

Declared compatible with :class:`MethodKind.FORMAL_VERIFICATION` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from ...contracts import MethodKind
from ..base_visualiser import BaseRobustnessVisualiser

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ...contracts import RobustnessVisualisationContext
    from ...results import RobustnessResult


_PLACEHOLDER_NO_BOUNDS = (
    "No output bounds present — configure `compute_output_bounds=True` on MarabouAssessor"
)
_PLACEHOLDER_NO_TARGETS = "OutputBoundsMarginHeatmapVisualiser requires per-sample targets"


def _placeholder_figure(message: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_axis_off()
    return fig


def _auto_figsize(n: int, k: int) -> tuple[float, float]:
    return (max(k * 0.6 + 2, 6), max(n * 0.25 + 1.5, 3))


class OutputBoundsMarginHeatmapVisualiser(BaseRobustnessVisualiser):
    """Heatmap of per-class lower-vs-upper margins relative to the target class."""

    supported_method_kinds: ClassVar[frozenset[MethodKind]] = frozenset(
        {MethodKind.FORMAL_VERIFICATION}
    )

    def __init__(
        self,
        *,
        cmap: str = "RdBu",
        max_samples: int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        self.cmap = cmap
        self.max_samples = int(max_samples) if max_samples is not None else None
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

        n, k = lower.shape
        if result.targets is None:
            return _placeholder_figure(_PLACEHOLDER_NO_TARGETS)
        targets_arr = np.asarray(result.targets.detach().cpu().numpy()).astype(int)
        if targets_arr.shape != (n,):
            return _placeholder_figure(_PLACEHOLDER_NO_TARGETS)

        if self.max_samples is not None and n > self.max_samples:
            lower = lower[: self.max_samples]
            upper = upper[: self.max_samples]
            targets_arr = targets_arr[: self.max_samples]
            n = self.max_samples

        target_lowers = lower[np.arange(n), targets_arr][:, None]  # shape (n, 1)
        margin = target_lowers - upper  # broadcast over k

        # Mask the target column for every row.
        mask = ~np.isfinite(margin)
        for i in range(n):
            t = targets_arr[i]
            if 0 <= t < k:
                mask[i, t] = True
        masked = np.ma.array(margin, mask=mask)

        if masked.compressed().size == 0:
            abs_max = 1.0
        else:
            abs_max = float(np.max(np.abs(masked.compressed())))
            if abs_max == 0.0:
                abs_max = 1.0
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

        figsize = self.figsize or _auto_figsize(n, k)
        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap(self.cmap).copy()
        cmap.set_bad(color="#bfbfbf")
        im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

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
        ax.set_title(f"{context.algorithm} — certified margin vs. target")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("lower[target] - upper[k]")
        fig.tight_layout()
        return fig
