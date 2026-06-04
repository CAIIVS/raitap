from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest
import torch

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

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
from raitap.robustness.exceptions import AssessmentKindVisualiserIncompatibilityError
from raitap.robustness.results import RobustnessMetrics, RobustnessResult, encode_verdicts
from raitap.robustness.visualisers import (
    ImagePairVisualiser,
    PerturbationHeatmapVisualiser,
)


def _image_axes(figure: Figure) -> list[Axes]:
    """Image panels only — drop the colorbar Axes matplotlib appends."""
    return [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]


def _make_result() -> RobustnessResult:
    inputs = torch.rand(2, 3, 8, 8)
    perturbed = (inputs + 0.05).clamp(0.0, 1.0)
    targets = torch.tensor([0, 1])
    return RobustnessResult(
        clean_inputs=inputs,
        targets=targets,
        clean_predictions=torch.tensor([0, 1]),
        verdicts=encode_verdicts(
            [RobustnessVerdict.ATTACK_SUCCEEDED, RobustnessVerdict.ATTACK_FAILED]
        ),
        metrics=RobustnessMetrics(clean_accuracy=1.0, adversarial_accuracy=0.5),
        run_dir=Path("."),
        experiment_name="t",
        adapter_target="raitap.robustness.assessors.TorchattacksAssessor",
        algorithm="PGD",
        name="pgd",
        perturbed_inputs=perturbed,
        perturbed_predictions=torch.tensor([1, 1]),
        perturbation_distance=torch.tensor([0.05, 0.05]),
        semantics=RobustnessSemantics(
            assessment_kind=AssessmentKind.EMPIRICAL_ATTACK,
            threat_model=ThreatModel.WHITE_BOX,
            objective=Objective.UNTARGETED,
            families={"gradient_sign"},
            perturbation=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        ),
    )


def _empirical_context() -> RobustnessVisualisationContext:
    return RobustnessVisualisationContext(
        algorithm="PGD",
        assessment_kind=AssessmentKind.EMPIRICAL_ATTACK,
        sample_names=["a", "b"],
        show_sample_names=True,
    )


def test_image_pair_visualiser_renders_figure() -> None:
    result = _make_result()
    visualiser = ImagePairVisualiser(max_samples=2)
    visualiser.validate_result(result)
    figure = visualiser.visualise(result, context=_empirical_context())
    try:
        assert len(_image_axes(figure)) == 6  # 2 rows by 3 columns (excl. colorbar)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("kwargs", "expected_axes", "expected_titles"),
    [
        ({}, 3, ["clean", "perturbed", "perturbation"]),
        ({"include_clean_input": False}, 2, ["perturbed", "perturbation"]),
        ({"include_perturbation_map": False}, 2, ["clean", "perturbed"]),
        (
            {"include_clean_input": False, "include_perturbation_map": False},
            1,
            ["perturbed"],
        ),
    ],
)
def test_image_pair_visualiser_honours_facet_kwargs(
    kwargs: dict[str, bool],
    expected_axes: int,
    expected_titles: list[str],
) -> None:
    result = _make_result()
    visualiser = ImagePairVisualiser(max_samples=1)

    figure = visualiser.visualise(result, context=_empirical_context(), **kwargs)

    try:
        image_axes = _image_axes(figure)
        assert len(image_axes) == expected_axes
        for axis, expected in zip(image_axes, expected_titles, strict=True):
            assert expected in axis.get_title()
    finally:
        plt.close(figure)


def test_empirical_visualisers_declare_embedded_facets() -> None:
    assert ImagePairVisualiser.embeds_clean_input is True
    assert ImagePairVisualiser.embeds_perturbation_map is True
    assert PerturbationHeatmapVisualiser.embeds_clean_input is False
    assert PerturbationHeatmapVisualiser.embeds_perturbation_map is True


def test_perturbation_heatmap_visualiser_renders_figure() -> None:
    result = _make_result()
    visualiser = PerturbationHeatmapVisualiser(max_samples=2)
    visualiser.validate_result(result)
    figure = visualiser.visualise(
        result,
        context=_empirical_context(),
        include_clean_input=False,
    )
    try:
        assert len(_image_axes(figure)) == 2
    finally:
        plt.close(figure)


def test_perturbation_heatmap_visualiser_rejects_render_without_its_facet() -> None:
    result = _make_result()
    visualiser = PerturbationHeatmapVisualiser(max_samples=1)

    with pytest.raises(ValueError, match="requires include_perturbation_map=True"):
        visualiser.visualise(
            result,
            context=_empirical_context(),
            include_perturbation_map=False,
        )


def test_image_pair_visualiser_rejects_non_image_modality() -> None:
    """ImagePairVisualiser must refuse tabular/text results, not silently feed garbage to imshow."""
    from raitap.transparency.contracts import InputKind, InputSpec

    result = _make_result()
    tabular_spec = InputSpec(kind=InputKind.TABULAR, shape=(2, 16), layout="(B,F)")
    object.__setattr__(
        result.semantics,
        "input_spec",
        tabular_spec,
    )
    visualiser = ImagePairVisualiser(max_samples=1)
    with pytest.raises(ValueError, match="image-modality results"):
        visualiser.visualise(result, context=_empirical_context())


def test_image_pair_visualiser_uses_shared_display_range_for_clean_and_perturbed() -> None:
    """Clean and perturbed must share a display scale; per-cell normalisation skews the eye."""
    inputs = torch.full((1, 3, 4, 4), 0.5)
    perturbed = inputs + 0.05  # within [0,1] after addition; both images sit in [0.5, 0.55]
    targets = torch.tensor([0])
    result = RobustnessResult(
        clean_inputs=inputs,
        targets=targets,
        clean_predictions=torch.tensor([0]),
        verdicts=encode_verdicts([RobustnessVerdict.ATTACK_FAILED]),
        metrics=RobustnessMetrics(clean_accuracy=1.0, adversarial_accuracy=1.0),
        run_dir=Path("."),
        experiment_name="t",
        adapter_target="raitap.robustness.assessors.TorchattacksAssessor",
        algorithm="FGSM",
        name="fgsm",
        perturbed_inputs=perturbed,
        perturbed_predictions=torch.tensor([0]),
        perturbation_distance=torch.tensor([0.05]),
        semantics=RobustnessSemantics(
            assessment_kind=AssessmentKind.EMPIRICAL_ATTACK,
            threat_model=ThreatModel.WHITE_BOX,
            objective=Objective.UNTARGETED,
            families={"gradient_sign"},
            perturbation=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        ),
    )
    visualiser = ImagePairVisualiser(max_samples=1)
    figure = visualiser.visualise(result, context=_empirical_context())
    try:
        clean_arr = figure.axes[0].images[0].get_array()
        perturbed_arr = figure.axes[1].images[0].get_array()
        assert clean_arr is not None and perturbed_arr is not None
        # Same display range -> images differ by exactly the perturbation, not by a stretched scale.
        assert clean_arr.shape == perturbed_arr.shape
        diff = abs(float(perturbed_arr.mean()) - float(clean_arr.mean()))
        # 0.05 perturbation under shared [0,1] range stays as 0.05 in display space.
        assert 0.04 < diff < 0.06
    finally:
        plt.close(figure)


def test_validate_result_blocks_wrong_assessment_kind() -> None:
    result = _make_result()
    # Force a verifier-only visualiser by patching supported_assessment_kinds.
    visualiser = ImagePairVisualiser(max_samples=1)
    type(visualiser).supported_assessment_kinds = frozenset({AssessmentKind.FORMAL_VERIFICATION})
    try:
        with pytest.raises(AssessmentKindVisualiserIncompatibilityError):
            visualiser.validate_result(result)
    finally:
        type(visualiser).supported_assessment_kinds = frozenset({AssessmentKind.EMPIRICAL_ATTACK})
