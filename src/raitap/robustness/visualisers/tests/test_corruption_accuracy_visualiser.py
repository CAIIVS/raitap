"""Content smoke test for the average-case visualiser — asserts structure, not pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

if TYPE_CHECKING:
    from pathlib import Path

matplotlib.use("Agg")

import torch
from matplotlib.container import ErrorbarContainer

from raitap.robustness.contracts import (
    AssessmentKind,
    Objective,
    PerturbationDistribution,
    RobustnessSemantics,
    RobustnessVisualisationContext,
    ThreatModel,
)
from raitap.robustness.results import RobustnessMetrics, RobustnessResult
from raitap.robustness.visualisers.average_case.corruption_accuracy_visualiser import (
    CorruptionAccuracyVisualiser,
)


def _make_result(tmp_path: Path) -> RobustnessResult:
    semantics = RobustnessSemantics(
        assessment_kind=AssessmentKind.STATISTICAL_SAMPLING,
        threat_model=ThreatModel.NOT_APPLICABLE,
        objective=Objective.UNTARGETED,
        families=frozenset({"noise"}),
        perturbation=PerturbationDistribution(corruption_name="gaussian_noise", severity=3),
    )
    return RobustnessResult(
        clean_inputs=torch.rand(4, 3, 8, 8),
        targets=torch.tensor([0, 1, 0, 1]),
        clean_predictions=torch.tensor([0, 1, 0, 1]),
        verdicts=torch.tensor([7, 7, 8, 7]),
        metrics=RobustnessMetrics(
            clean_accuracy=1.0,
            corrupted_accuracy=0.75,
            accuracy_ci_low=0.3,
            accuracy_ci_high=0.95,
            n_samples=4,
            n_correct=3,
        ),
        run_dir=tmp_path,
        experiment_name="t",
        adapter_target="x",
        algorithm="gaussian_noise",
        semantics=semantics,
    )


def test_visualiser_renders_two_bars_with_error_bar(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    vis = CorruptionAccuracyVisualiser()
    context = RobustnessVisualisationContext(
        algorithm="gaussian_noise",
        assessment_kind=AssessmentKind.STATISTICAL_SAMPLING,
        sample_names=None,
        show_sample_names=False,
    )
    vis.validate_result(result)
    fig = vis.visualise(result, context=context)

    ax = fig.axes[0]
    bars = list(ax.patches)
    assert len(bars) == 2  # clean + corrupted
    # CI whisker must be present — assert an actual ErrorbarContainer, not just
    # any container (a lone bar container would satisfy a >=1 check and hide a
    # regression that drops the errorbar).
    assert any(isinstance(c, ErrorbarContainer) for c in ax.containers)
    assert AssessmentKind.STATISTICAL_SAMPLING in vis.supported_assessment_kinds
