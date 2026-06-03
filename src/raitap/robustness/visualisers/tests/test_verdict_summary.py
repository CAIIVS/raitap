"""Snapshot-style test for :class:`VerdictSummaryVisualiser`."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from raitap.robustness.contracts import (
    AssessmentKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessSemantics,
    RobustnessVerdict,
    RobustnessVisualisationContext,
    ThreatModel,
)
from raitap.robustness.results import RobustnessMetrics, RobustnessResult, encode_verdicts
from raitap.robustness.visualisers import VerdictSummaryVisualiser


def _make_formal_result() -> RobustnessResult:
    inputs = torch.zeros(4, 5)
    targets = torch.tensor([0, 1, 2, 3])
    verdicts = [
        RobustnessVerdict.VERIFIED,
        RobustnessVerdict.FALSIFIED,
        RobustnessVerdict.UNKNOWN,
        RobustnessVerdict.ERROR,
    ]
    return RobustnessResult(
        clean_inputs=inputs,
        targets=targets,
        clean_predictions=targets.clone(),
        verdicts=encode_verdicts(verdicts),
        metrics=RobustnessMetrics(
            clean_accuracy=1.0,
            verified_rate=0.25,
            falsified_rate=0.25,
            unknown_rate=0.25,
            error_rate=0.25,
            mean_runtime=0.5,
        ),
        run_dir=Path("."),
        experiment_name="marabou-test",
        adapter_target="raitap.robustness.assessors.MarabouAssessor",
        algorithm="linf-box",
        name="marabou_linf",
        runtime_per_sample=torch.tensor([0.1, 0.4, 0.7, 0.8]),
        semantics=RobustnessSemantics(
            assessment_kind=AssessmentKind.FORMAL_VERIFICATION,
            threat_model=ThreatModel.WHITE_BOX,
            objective=Objective.UNTARGETED,
            families=frozenset({"smt", "complete", "sound"}),
            perturbation=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        ),
    )


def _formal_context() -> RobustnessVisualisationContext:
    return RobustnessVisualisationContext(
        algorithm="linf-box",
        assessment_kind=AssessmentKind.FORMAL_VERIFICATION,
        sample_names=None,
        show_sample_names=False,
    )


def test_verdict_summary_renders_two_panels() -> None:
    visualiser = VerdictSummaryVisualiser()
    result = _make_formal_result()
    visualiser.validate_result(result)
    figure = visualiser.visualise(result, context=_formal_context())
    try:
        assert len(figure.axes) == 2
        bar_ax, hist_ax = figure.axes
        assert bar_ax.get_ylabel() == "count"
        assert hist_ax.get_xlabel() == "seconds"
    finally:
        plt.close(figure)


def test_verdict_summary_only_supports_formal_verification() -> None:
    visualiser = VerdictSummaryVisualiser()
    assert AssessmentKind.FORMAL_VERIFICATION in visualiser.supported_assessment_kinds
    assert AssessmentKind.EMPIRICAL_ATTACK not in visualiser.supported_assessment_kinds
