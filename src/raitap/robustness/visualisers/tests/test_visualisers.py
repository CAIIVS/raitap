from __future__ import annotations

from pathlib import Path

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
from raitap.robustness.exceptions import MethodKindVisualiserIncompatibilityError
from raitap.robustness.results import RobustnessMetrics, RobustnessResult, encode_verdicts
from raitap.robustness.visualisers import (
    ImagePairVisualiser,
    PerturbationHeatmapVisualiser,
)


def _make_result() -> RobustnessResult:
    inputs = torch.rand(2, 3, 8, 8)
    perturbed = (inputs + 0.05).clamp(0.0, 1.0)
    targets = torch.tensor([0, 1])
    return RobustnessResult(
        clean_inputs=inputs,
        targets=targets,
        clean_predictions=torch.tensor([0, 1]),
        verdicts=encode_verdicts([RobustnessVerdict.ATTACKED, RobustnessVerdict.NOT_ATTACKED]),
        metrics=RobustnessMetrics(clean_accuracy=1.0, adversarial_accuracy=0.5),
        run_dir=Path("."),
        experiment_name="t",
        assessor_target="raitap.robustness.assessors.TorchattacksAssessor",
        algorithm="PGD",
        assessor_name="pgd",
        perturbed_inputs=perturbed,
        perturbed_predictions=torch.tensor([1, 1]),
        perturbation_distance=torch.tensor([0.05, 0.05]),
        semantics=RobustnessSemantics(
            method_kind=MethodKind.EMPIRICAL_ATTACK,
            threat_model=ThreatModel.WHITE_BOX,
            objective=Objective.UNTARGETED,
            families=frozenset({"gradient_sign"}),
            budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        ),
    )


def _empirical_context() -> RobustnessVisualisationContext:
    return RobustnessVisualisationContext(
        algorithm="PGD",
        method_kind=MethodKind.EMPIRICAL_ATTACK,
        sample_names=["a", "b"],
        show_sample_names=True,
    )


def test_image_pair_visualiser_renders_figure() -> None:
    result = _make_result()
    visualiser = ImagePairVisualiser(max_samples=2)
    visualiser.validate_result(result)
    figure = visualiser.visualise(result, context=_empirical_context())
    try:
        assert len(figure.axes) == 6  # 2 rows by 3 columns
    finally:
        plt.close(figure)


def test_perturbation_heatmap_visualiser_renders_figure() -> None:
    result = _make_result()
    visualiser = PerturbationHeatmapVisualiser(max_samples=2)
    visualiser.validate_result(result)
    figure = visualiser.visualise(result, context=_empirical_context())
    try:
        assert len(figure.axes) == 2
    finally:
        plt.close(figure)


def test_validate_result_blocks_wrong_method_kind() -> None:
    result = _make_result()
    # Force a verifier-only visualiser by patching supported_method_kinds.
    visualiser = ImagePairVisualiser(max_samples=1)
    type(visualiser).supported_method_kinds = frozenset({MethodKind.FORMAL_VERIFICATION})
    try:
        with pytest.raises(MethodKindVisualiserIncompatibilityError):
            visualiser.validate_result(result)
    finally:
        type(visualiser).supported_method_kinds = frozenset({MethodKind.EMPIRICAL_ATTACK})
