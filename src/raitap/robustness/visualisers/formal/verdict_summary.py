"""Summary visualiser for formal-verification robustness results.

Two side-by-side panels:

* Left: bar chart of per-verdict counts (VERIFIED / FALSIFIED / UNKNOWN /
  ERROR), colour-coded so the reader can tell at a glance how the verifier
  performed across the batch.
* Right: histogram of ``runtime_per_sample`` (seconds), to surface
  long-tail timeouts vs. fast SAT/UNSAT cases.

Declared compatible with :class:`AssessmentKind.FORMAL_VERIFICATION` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from raitap.robustness.visualisers.registration import robustness_visualiser

from ...contracts import AssessmentKind, ReportFigureScope, RobustnessVerdict
from ..base_visualiser import BaseRobustnessVisualiser

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ...contracts import RobustnessVisualisationContext
    from ...results import RobustnessResult


_VERDICT_ORDER = (
    RobustnessVerdict.VERIFIED,
    RobustnessVerdict.FALSIFIED,
    RobustnessVerdict.UNKNOWN,
    RobustnessVerdict.ERROR,
)
_VERDICT_COLORS = {
    RobustnessVerdict.VERIFIED: "#2ca02c",
    RobustnessVerdict.FALSIFIED: "#d62728",
    RobustnessVerdict.UNKNOWN: "#7f7f7f",
    RobustnessVerdict.ERROR: "#9467bd",
}


@robustness_visualiser(
    registry_name="verdict_summary",
    supported_assessment_kinds={AssessmentKind.FORMAL_VERIFICATION},
    report_figure_scope=ReportFigureScope.ASSESSOR,
)
class VerdictSummaryVisualiser(BaseRobustnessVisualiser):
    """Bar chart of verdict counts plus a runtime histogram."""

    def __init__(self, *, runtime_bins: int = 20) -> None:
        self.runtime_bins = max(int(runtime_bins), 1)

    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        del kwargs
        from ...results import decode_verdicts

        verdicts = decode_verdicts(result.verdicts)
        counts: dict[RobustnessVerdict, int] = dict.fromkeys(_VERDICT_ORDER, 0)
        for v in verdicts:
            if v in counts:
                counts[v] += 1

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        labels = [v.value for v in _VERDICT_ORDER]
        values = [counts[v] for v in _VERDICT_ORDER]
        colors = [_VERDICT_COLORS[v] for v in _VERDICT_ORDER]
        axes[0].bar(labels, values, color=colors)
        axes[0].set_title("Verdict distribution")
        axes[0].set_xlabel("Verdict")
        axes[0].set_ylabel("Count")
        for index, count in enumerate(values):
            axes[0].text(index, count, str(count), ha="center", va="bottom", fontsize=9)

        runtimes = result.runtime_per_sample
        if runtimes is not None and runtimes.numel():
            data = np.asarray(runtimes.detach().cpu().numpy(), dtype=float)
            axes[1].hist(data, bins=self.runtime_bins, color="#1f77b4")
            axes[1].set_title("Runtime per sample")
            axes[1].set_xlabel("Seconds")
            axes[1].set_ylabel("Samples")
        else:
            axes[1].text(
                0.5,
                0.5,
                "no runtime data",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_axis_off()

        fig.suptitle(f"{context.algorithm} — formal-verification summary", fontsize=12)
        fig.tight_layout()
        return fig
