from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest
import torch

from raitap.robustness.contracts import (
    MethodKind,
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
from raitap.robustness.visualisers.base_visualiser import (
    BaseRobustnessVisualiser,
    _RobustnessVisualisationSkipped,
)

if TYPE_CHECKING:
    from typing import Any

    from matplotlib.figure import Figure


def _semantics_for_test() -> RobustnessSemantics:
    return RobustnessSemantics(
        method_kind=MethodKind.EMPIRICAL_ATTACK,
        threat_model=ThreatModel.WHITE_BOX,
        objective=Objective.UNTARGETED,
        families=frozenset({"gradient_sign"}),
        budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.03),
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
    verdicts = encode_verdicts([RobustnessVerdict.ATTACKED, RobustnessVerdict.NOT_ATTACKED])

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
    assert metadata["method_kind"] == "empirical_attack"
    assert metadata["semantics"]["budget"]["norm"] == "Linf"
    assert metadata["metrics"]["attack_success_rate"] == 0.75
    assert metadata["verdict_codes"]["attacked"] == 1


def _result_for_visualiser_tests(tmp_path: Path) -> RobustnessResult:
    inputs = torch.randn(2, 3, 4, 4)
    return RobustnessResult(
        clean_inputs=inputs,
        targets=torch.tensor([0, 1]),
        clean_predictions=torch.tensor([0, 1]),
        verdicts=encode_verdicts([RobustnessVerdict.ATTACKED, RobustnessVerdict.NOT_ATTACKED]),
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


class _SkippingRobustnessVisualiser(BaseRobustnessVisualiser):
    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        del result, context, kwargs
        raise _RobustnessVisualisationSkipped("skip this report-only render")


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
    result.verdicts = encode_verdicts([RobustnessVerdict.NOT_ATTACKED, RobustnessVerdict.ATTACKED])
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


def test_render_visualisation_for_report_skip_has_no_side_effects(tmp_path: Path) -> None:
    result = _result_for_visualiser_tests(tmp_path)
    result.visualiser_targets.append("existing.Visualiser_0")
    result.visualisers = [
        ConfiguredRobustnessVisualiser(visualiser=_SkippingRobustnessVisualiser())
    ]

    rendered = result.render_visualisation_for_report(0)

    assert rendered is None
    assert result.visualiser_targets == ["existing.Visualiser_0"]
    assert not result.run_dir.exists()


def test_render_visualisation_for_report_out_of_range_raises_index_error(
    tmp_path: Path,
) -> None:
    result = _result_for_visualiser_tests(tmp_path)

    with pytest.raises(IndexError):
        result.render_visualisation_for_report(0)


def test_visualise_persists_pngs_and_does_not_swallow_skip_exception(tmp_path: Path) -> None:
    result = _result_for_visualiser_tests(tmp_path)
    result.visualisers = [
        ConfiguredRobustnessVisualiser(visualiser=_SkippingRobustnessVisualiser())
    ]

    with pytest.raises(_RobustnessVisualisationSkipped):
        result.visualise()

    assert result.visualiser_targets == []
    assert not any(result.run_dir.glob("*.png"))
