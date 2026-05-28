from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest
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
from raitap.robustness.results import (
    ConfiguredRobustnessVisualiser,
    RobustnessMetrics,
    RobustnessResult,
    encode_verdicts,
)
from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser

if TYPE_CHECKING:
    from typing import Any

    from matplotlib.figure import Figure


def _semantics_for_test() -> RobustnessSemantics:
    return RobustnessSemantics(
        assessment_kind=AssessmentKind.EMPIRICAL_ATTACK,
        threat_model=ThreatModel.WHITE_BOX,
        objective=Objective.UNTARGETED,
        families=frozenset({"gradient_sign"}),
        perturbation=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.03),
    )


def _empirical_metrics() -> RobustnessMetrics:
    return RobustnessMetrics(
        clean_accuracy=0.9,
        adversarial_accuracy=0.1,
        attack_success_rate=0.75,
        mean_distance=0.02,
        max_distance=0.03,
    )


def test_metrics_as_dict_drops_none_fields() -> None:
    metrics = _empirical_metrics()
    out = metrics.as_dict()
    assert out["clean_accuracy"] == 0.9
    assert "adversarial_accuracy" in out
    assert "verified_rate" not in out


def test_robustness_result_writes_pt_and_metadata(tmp_path: Path) -> None:
    inputs = torch.randn(2, 3, 4, 4)
    perturbed = inputs + 0.01
    targets = torch.tensor([0, 1])
    clean_preds = torch.tensor([0, 1])
    adv_preds = torch.tensor([1, 1])
    verdicts = encode_verdicts(
        [RobustnessVerdict.ATTACK_SUCCEEDED, RobustnessVerdict.ATTACK_FAILED]
    )

    result = RobustnessResult(
        clean_inputs=inputs,
        targets=targets,
        clean_predictions=clean_preds,
        verdicts=verdicts,
        metrics=_empirical_metrics(),
        run_dir=tmp_path / "robustness/pgd",
        experiment_name="unit-test",
        assessor_target="raitap.robustness.assessors.TorchattacksAssessor",
        algorithm="PGD",
        assessor_name="pgd",
        perturbed_inputs=perturbed,
        perturbed_predictions=adv_preds,
        perturbation_distance=torch.tensor([0.02, 0.02]),
        semantics=_semantics_for_test(),
    )
    result.write_artifacts()

    payload = torch.load(result.run_dir / "robustness_data.pt", weights_only=False)
    assert "perturbed_inputs" in payload
    assert payload["verdicts"].tolist() == [1, 2]

    metadata = json.loads((result.run_dir / "metadata.json").read_text())
    assert metadata["assessment_kind"] == "empirical_attack"
    assert metadata["semantics"]["perturbation"]["norm"] == "Linf"
    assert metadata["metrics"]["attack_success_rate"] == 0.75
    assert metadata["verdict_codes"]["attack_succeeded"] == 1


def _result_for_visualiser_tests(tmp_path: Path) -> RobustnessResult:
    inputs = torch.randn(2, 3, 4, 4)
    return RobustnessResult(
        clean_inputs=inputs,
        targets=torch.tensor([0, 1]),
        clean_predictions=torch.tensor([0, 1]),
        verdicts=encode_verdicts(
            [RobustnessVerdict.ATTACK_SUCCEEDED, RobustnessVerdict.ATTACK_FAILED]
        ),
        metrics=_empirical_metrics(),
        run_dir=tmp_path / "robustness/pgd",
        experiment_name="unit-test",
        assessor_target="raitap.robustness.assessors.TorchattacksAssessor",
        algorithm="PGD",
        assessor_name="pgd",
        perturbed_inputs=inputs + 0.01,
        perturbed_predictions=torch.tensor([1, 1]),
        perturbation_distance=torch.tensor([0.02, 0.02]),
        semantics=_semantics_for_test(),
    )


class _RecordingRobustnessVisualiser(BaseRobustnessVisualiser):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result_shapes: list[tuple[int, ...]] = []
        self.targets_seen: list[list[int]] = []
        self.context_sample_names: list[list[str] | None] = []

    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        self.calls.append(dict(kwargs))
        self.result_shapes.append(tuple(result.clean_inputs.shape))
        self.targets_seen.append([int(item) for item in result.targets.tolist()])
        self.context_sample_names.append(context.sample_names)
        fig, _ax = plt.subplots(figsize=(1, 1))
        return fig


class _ErrorRobustnessVisualiser(BaseRobustnessVisualiser):
    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        del result, context, kwargs
        raise ValueError("visualiser failed")


def test_render_visualisation_for_report_targets_one_visualiser_and_forwards_kwargs(
    tmp_path: Path,
) -> None:
    first = _RecordingRobustnessVisualiser()
    second = _RecordingRobustnessVisualiser()
    result = _result_for_visualiser_tests(tmp_path)
    result.visualisers = [
        ConfiguredRobustnessVisualiser(visualiser=first, call_kwargs={"alpha": 0.1}),
        ConfiguredRobustnessVisualiser(visualiser=second, call_kwargs={"alpha": 0.2}),
    ]

    rendered = result.render_visualisation_for_report(
        1,
        alpha=0.9,
        include_perturbation_map=False,
    )

    assert rendered is not None
    assert rendered.visualiser_name == "_RecordingRobustnessVisualiser_1"
    assert rendered.output_path == Path("_RecordingRobustnessVisualiser_1.png")
    assert first.calls == []
    assert second.calls == [{"alpha": 0.9, "include_perturbation_map": False}]
    assert result.visualiser_targets == []
    assert not result.run_dir.exists()
    plt.close(rendered.figure)


def test_render_visualisation_for_report_slices_to_requested_sample(
    tmp_path: Path,
) -> None:
    visualiser = _RecordingRobustnessVisualiser()
    result = _result_for_visualiser_tests(tmp_path)
    result.clean_inputs = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    result.perturbed_inputs = result.clean_inputs + 0.5
    result.targets = torch.tensor([4, 7])
    result.clean_predictions = torch.tensor([4, 5])
    result.perturbed_predictions = torch.tensor([6, 8])
    result.verdicts = encode_verdicts(
        [RobustnessVerdict.ATTACK_FAILED, RobustnessVerdict.ATTACK_SUCCEEDED]
    )
    result.perturbation_distance = torch.tensor([0.1, 0.2])
    result.runtime_per_sample = torch.tensor([1.5, 2.5])
    result.output_bounds = {
        "lower": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        "global": torch.tensor([9.0]),
    }
    result.kwargs["sample_names"] = ["sample-a", "sample-b"]
    result.kwargs["show_sample_names"] = True
    result.visualisers = [ConfiguredRobustnessVisualiser(visualiser=visualiser)]

    rendered = result.render_visualisation_for_report(0, sample_index=1)

    assert rendered is not None
    assert visualiser.result_shapes == [(1, 3, 4, 4)]
    assert visualiser.targets_seen == [[7]]
    assert visualiser.context_sample_names == [["sample-b"]]
    assert result.clean_inputs.shape == (2, 3, 4, 4)
    assert result.visualiser_targets == []
    assert not result.run_dir.exists()
    plt.close(rendered.figure)


def test_render_visualisation_for_report_sample_index_out_of_range_raises(
    tmp_path: Path,
) -> None:
    result = _result_for_visualiser_tests(tmp_path)
    result.visualisers = [
        ConfiguredRobustnessVisualiser(visualiser=_RecordingRobustnessVisualiser())
    ]

    with pytest.raises(IndexError, match="sample_index"):
        result.render_visualisation_for_report(0, sample_index=2)


def test_render_visualisation_for_report_out_of_range_raises_index_error(
    tmp_path: Path,
) -> None:
    result = _result_for_visualiser_tests(tmp_path)

    with pytest.raises(IndexError):
        result.render_visualisation_for_report(0)


def test_visualise_does_not_swallow_visualiser_errors(tmp_path: Path) -> None:
    result = _result_for_visualiser_tests(tmp_path)
    result.visualisers = [ConfiguredRobustnessVisualiser(visualiser=_ErrorRobustnessVisualiser())]

    with pytest.raises(ValueError, match="visualiser failed"):
        result.visualise()

    assert result.visualiser_targets == []
    assert not any(result.run_dir.glob("*.png"))


# ---------------------------------------------------------------------------
# C3 / C4 — RobustnessResult sample_names length-validation tests
# ---------------------------------------------------------------------------


def _result_with_sample_names(
    tmp_path: Path,
    *,
    batch: int,
    sample_names: list[str] | None,
) -> RobustnessResult:
    """Build a RobustnessResult with a recording visualiser and the given sample_names."""
    inputs = torch.zeros(batch, 3, 4, 4)
    result = RobustnessResult(
        clean_inputs=inputs,
        targets=torch.zeros(batch, dtype=torch.long),
        clean_predictions=torch.zeros(batch, dtype=torch.long),
        verdicts=encode_verdicts([RobustnessVerdict.ATTACK_FAILED] * batch),
        metrics=_empirical_metrics(),
        run_dir=tmp_path / "robustness/pgd",
        experiment_name="test_sample_names",
        assessor_target="raitap.robustness.assessors.TorchattacksAssessor",
        algorithm="PGD",
        assessor_name="pgd",
        semantics=_semantics_for_test(),
    )
    result.kwargs["sample_names"] = sample_names
    result.visualisers = [
        ConfiguredRobustnessVisualiser(visualiser=_RecordingRobustnessVisualiser())
    ]
    return result


def test_robustness_visualise_raises_on_mismatched_sample_names(tmp_path: Path) -> None:
    result = _result_with_sample_names(tmp_path, batch=2, sample_names=["a", "b", "c"])
    from raitap.utils.errors import SampleNamesLengthError

    with pytest.raises(SampleNamesLengthError) as info:
        result.visualise()
    assert info.value.got == 3
    assert info.value.expected == 2


def test_robustness_render_raises_on_mismatched_sample_names(tmp_path: Path) -> None:
    result = _result_with_sample_names(tmp_path, batch=2, sample_names=["a", "b", "c"])
    from raitap.utils.errors import SampleNamesLengthError

    with pytest.raises(SampleNamesLengthError):
        result.render_visualisation_for_report(0)  # full-batch path (sample_index=None)


def test_robustness_render_per_sample_raises_on_mismatched_sample_names(tmp_path: Path) -> None:
    # batch=3, sample_names length=2 → mismatch should raise even on valid sample_index.
    result = _result_with_sample_names(tmp_path, batch=3, sample_names=["a", "b"])
    from raitap.utils.errors import SampleNamesLengthError

    with pytest.raises(SampleNamesLengthError):
        result.render_visualisation_for_report(0, sample_index=2)


def test_robustness_visualise_passes_with_matching_sample_names(tmp_path: Path) -> None:
    result = _result_with_sample_names(tmp_path, batch=2, sample_names=["a", "b"])
    # Must not raise; close the figure to avoid resource leak.
    vis_results = result.visualise()
    for vr in vis_results:
        plt.close(vr.figure)


def test_robustness_visualise_passes_with_none_sample_names(tmp_path: Path) -> None:
    result = _result_with_sample_names(tmp_path, batch=2, sample_names=None)
    vis_results = result.visualise()
    for vr in vis_results:
        plt.close(vr.figure)


def test_metrics_average_case_fields_in_as_dict() -> None:
    from raitap.robustness.results import RobustnessMetrics

    metrics = RobustnessMetrics(
        clean_accuracy=0.9,
        corrupted_accuracy=0.7,
        accuracy_ci_low=0.6,
        accuracy_ci_high=0.8,
        n_samples=10,
        n_correct=7,
    )
    out = metrics.as_dict()
    assert out["corrupted_accuracy"] == 0.7
    assert out["accuracy_ci_low"] == 0.6
    assert out["accuracy_ci_high"] == 0.8
    assert out["n_samples"] == 10.0
    assert out["n_correct"] == 7.0
    # Worst-case fields stay absent when None.
    assert "adversarial_accuracy" not in out


def test_metadata_includes_case(tmp_path: Path) -> None:
    from raitap.robustness.contracts import PerturbationDistribution

    semantics = RobustnessSemantics(
        assessment_kind=AssessmentKind.STATISTICAL_SAMPLING,
        threat_model=ThreatModel.NOT_APPLICABLE,
        objective=Objective.UNTARGETED,
        families=frozenset({"noise"}),
        perturbation=PerturbationDistribution(corruption_name="fog", severity=2),
    )
    result = RobustnessResult(
        clean_inputs=torch.rand(2, 3, 8, 8),
        targets=torch.tensor([0, 1]),
        clean_predictions=torch.tensor([0, 1]),
        verdicts=torch.tensor([7, 8]),
        metrics=RobustnessMetrics(
            clean_accuracy=1.0, corrupted_accuracy=0.5, n_samples=2, n_correct=1
        ),
        run_dir=str(tmp_path),
        experiment_name="t",
        assessor_target="x",
        algorithm="fog",
        semantics=semantics,
    )
    meta = result._metadata()
    assert meta["case"] == "average_case"
    assert meta["assessment_kind"] == "statistical_sampling"
