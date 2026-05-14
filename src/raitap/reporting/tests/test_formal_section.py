"""Reporting section assertions for formal-verification robustness results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

from raitap.configs import set_output_root
from raitap.configs.schema import AppConfig, ReportingConfig
from raitap.pipeline.outputs import RunOutputs
from raitap.reporting.builder import build_report
from raitap.robustness.contracts import (
    MethodKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessSemantics,
    RobustnessVerdict,
    ThreatModel,
)
from raitap.robustness.results import RobustnessMetrics, RobustnessResult, encode_verdicts


def _formal_result(run_dir: Path) -> RobustnessResult:
    inputs = torch.zeros(3, 5)
    targets = torch.tensor([0, 1, 2])
    return RobustnessResult(
        clean_inputs=inputs,
        targets=targets,
        clean_predictions=targets.clone(),
        verdicts=encode_verdicts(
            [
                RobustnessVerdict.VERIFIED,
                RobustnessVerdict.FALSIFIED,
                RobustnessVerdict.UNKNOWN,
            ]
        ),
        metrics=RobustnessMetrics(
            clean_accuracy=1.0,
            verified_rate=1 / 3,
            falsified_rate=1 / 3,
            unknown_rate=1 / 3,
            error_rate=0.0,
            mean_runtime=0.5,
        ),
        run_dir=run_dir,
        experiment_name="exp",
        assessor_target="raitap.robustness.assessors.MarabouAssessor",
        algorithm="linf-box",
        assessor_name="marabou_linf",
        runtime_per_sample=torch.tensor([0.2, 0.4, 0.6]),
        semantics=RobustnessSemantics(
            method_kind=MethodKind.FORMAL_VERIFICATION,
            threat_model=ThreatModel.WHITE_BOX,
            objective=Objective.UNTARGETED,
            families=frozenset({"smt", "complete", "sound"}),
            budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        ),
    )


def test_build_report_renders_robustness_certification_section(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="marabou_test")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    run_dir = tmp_path / "robustness" / "marabou_linf"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = _formal_result(run_dir)

    outputs = RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=torch.zeros(3, 5),
        sample_ids=None,
        robustness_results=[result],
        robustness_visualisations=[],
    )

    report = build_report(config, outputs)
    robustness_sections = [s for s in report.sections if s.title == "Robustness"]
    assert len(robustness_sections) == 1
    section = robustness_sections[0]
    assert len(section.groups) == 1
    group = section.groups[0]
    assert group.heading.startswith("Robustness certification - linf-box")
    row_keys = {key for key, _ in group.table_rows}
    assert {"verified_rate", "falsified_rate", "unknown_rate"}.issubset(row_keys)


def test_build_report_includes_per_class_bound_rows_for_formal_results(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="marabou_test")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    run_dir = tmp_path / "robustness" / "marabou_linf"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = _formal_result(run_dir)
    # Two verified samples + one NaN row → 3 samples, 5 classes.
    result.output_bounds = {
        "lower": torch.tensor(
            [
                [-1.0, -0.5, -0.3, -0.2, -0.1],
                [-2.0, -1.0, -0.6, -0.4, -0.2],
                [float("nan")] * 5,
            ]
        ),
        "upper": torch.tensor(
            [
                [1.0, 0.5, 0.3, 0.2, 0.1],
                [2.0, 1.0, 0.6, 0.4, 0.2],
                [float("nan")] * 5,
            ]
        ),
    }

    outputs = RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=torch.zeros(3, 5),
        sample_ids=None,
        robustness_results=[result],
        robustness_visualisations=[],
    )

    report = build_report(config, outputs)
    section = next(s for s in report.sections if s.title == "Robustness")
    rows = dict(section.groups[0].table_rows)
    assert rows["logit_0_lower_mean"] == f"{(-1.0 + -2.0) / 2:.4f}"
    assert rows["logit_0_upper_mean"] == f"{(1.0 + 2.0) / 2:.4f}"
    assert rows["logit_4_lower_mean"] == f"{(-0.1 + -0.2) / 2:.4f}"
    assert rows["output_bounds_samples"] == "2/3"


def test_build_report_excludes_rows_with_only_lower_bound(tmp_path: Path) -> None:
    """A row with all-NaN upper must NOT count toward output_bounds_samples."""
    config = AppConfig(experiment_name="marabou_test")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")

    run_dir = tmp_path / "robustness" / "marabou_linf"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = _formal_result(run_dir)
    # Row 0: both sides present. Row 1: lower present, upper all NaN. Row 2: NaN row.
    result.output_bounds = {
        "lower": torch.tensor(
            [
                [-1.0, -0.5, -0.3, -0.2, -0.1],
                [-2.0, -1.0, -0.6, -0.4, -0.2],
                [float("nan")] * 5,
            ]
        ),
        "upper": torch.tensor(
            [
                [1.0, 0.5, 0.3, 0.2, 0.1],
                [float("nan")] * 5,
                [float("nan")] * 5,
            ]
        ),
    }

    outputs = RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=torch.zeros(3, 5),
        sample_ids=None,
        robustness_results=[result],
        robustness_visualisations=[],
    )

    report = build_report(config, outputs)
    section = next(s for s in report.sections if s.title == "Robustness")
    rows = dict(section.groups[0].table_rows)
    # Only row 0 has BOTH sides non-NaN → 1/3.
    assert rows["output_bounds_samples"] == "1/3"


def test_build_report_skips_bound_rows_for_empirical_attack_results(
    tmp_path: Path,
) -> None:
    config = AppConfig(experiment_name="ea_test")
    set_output_root(config, tmp_path)
    config.reporting = ReportingConfig(_target_="PDFReporter", filename="report.pdf")
    run_dir = tmp_path / "robustness" / "ea"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = _formal_result(run_dir)
    # Mutate to an EMPIRICAL_ATTACK result while keeping output_bounds set.
    result.semantics = RobustnessSemantics(
        method_kind=MethodKind.EMPIRICAL_ATTACK,
        threat_model=ThreatModel.WHITE_BOX,
        objective=Objective.UNTARGETED,
        families=frozenset({"attack"}),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
    )
    result.output_bounds = {
        "lower": torch.zeros(3, 5),
        "upper": torch.ones(3, 5),
    }

    outputs = RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        forward_output=torch.zeros(3, 5),
        sample_ids=None,
        robustness_results=[result],
        robustness_visualisations=[],
    )

    report = build_report(config, outputs)
    section = next(s for s in report.sections if s.title == "Robustness")
    rows = dict(section.groups[0].table_rows)
    assert "logit_0_lower_mean" not in rows
    assert "output_bounds_samples" not in rows
