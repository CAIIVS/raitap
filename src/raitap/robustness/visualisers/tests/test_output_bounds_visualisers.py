"""Tests for the formal-verification output-bounds visualisers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    OutputBoundsCohortVisualiser,
    OutputBoundsMarginHeatmapVisualiser,
    OutputBoundsPinnedVisualiser,
    OutputBoundsWidthHeatmapVisualiser,
)


def _formal_result(
    *,
    n: int = 5,
    k: int = 4,
    output_bounds: Any = "default",
) -> RobustnessResult:
    inputs = torch.zeros(n, 3)
    targets = torch.arange(n) % k
    verdicts = [RobustnessVerdict.VERIFIED] * n

    if output_bounds == "default":
        lower = torch.full((n, k), float("nan"))
        upper = torch.full((n, k), float("nan"))
        for i in range(n):
            for j in range(k):
                lower[i, j] = -1.0 - 0.1 * j
                upper[i, j] = 1.0 + 0.1 * j + 0.05 * i
        output_bounds = {"lower": lower, "upper": upper}

    return RobustnessResult(
        clean_inputs=inputs,
        targets=targets,
        clean_predictions=targets.clone(),
        verdicts=encode_verdicts(verdicts),
        metrics=RobustnessMetrics(clean_accuracy=1.0, verified_rate=1.0),
        run_dir=Path("."),
        experiment_name="marabou-test",
        assessor_target="raitap.robustness.assessors.MarabouAssessor",
        algorithm="linf-box",
        assessor_name="marabou_linf",
        output_bounds=output_bounds,
        runtime_per_sample=torch.zeros(n),
        semantics=RobustnessSemantics(
            method_kind=MethodKind.FORMAL_VERIFICATION,
            threat_model=ThreatModel.WHITE_BOX,
            objective=Objective.UNTARGETED,
            families=frozenset({"smt"}),
            budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        ),
    )


def _empirical_result() -> RobustnessResult:
    inputs = torch.zeros(2, 3)
    targets = torch.tensor([0, 1])
    verdicts = [RobustnessVerdict.ATTACKED, RobustnessVerdict.NOT_ATTACKED]
    return RobustnessResult(
        clean_inputs=inputs,
        targets=targets,
        clean_predictions=targets.clone(),
        verdicts=encode_verdicts(verdicts),
        metrics=RobustnessMetrics(clean_accuracy=1.0, adversarial_accuracy=0.5),
        run_dir=Path("."),
        experiment_name="pgd-test",
        assessor_target="raitap.robustness.assessors.TorchattacksAssessor",
        algorithm="PGD",
        semantics=RobustnessSemantics(
            method_kind=MethodKind.EMPIRICAL_ATTACK,
            threat_model=ThreatModel.WHITE_BOX,
            objective=Objective.UNTARGETED,
            families=frozenset({"gradient"}),
            budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        ),
    )


def _ctx(
    *, show_sample_names: bool = False, names: list[str] | None = None
) -> RobustnessVisualisationContext:
    return RobustnessVisualisationContext(
        algorithm="linf-box",
        method_kind=MethodKind.FORMAL_VERIFICATION,
        sample_names=names,
        show_sample_names=show_sample_names,
    )


# ---------------------------- cohort ---------------------------------------


def test_cohort_visualiser_renders_box_per_class() -> None:
    visualiser = OutputBoundsCohortVisualiser()
    result = _formal_result(n=5, k=4)
    figure = visualiser.visualise(result, context=_ctx())
    try:
        assert len(figure.axes) == 1
        ax = figure.axes[0]
        assert len(ax.get_xticklabels()) == 4
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == [f"logit_{k}" for k in range(4)]
        # boxplot creates Line2D artists for whiskers/caps/medians; just assert non-empty.
        assert ax.lines, "expected boxplot artists on the axis"
    finally:
        plt.close(figure)


def test_cohort_visualiser_handles_none_bounds_gracefully() -> None:
    visualiser = OutputBoundsCohortVisualiser()
    result = _formal_result(output_bounds=None)
    figure = visualiser.visualise(result, context=_ctx())
    try:
        assert len(figure.axes) == 1
        texts = [t.get_text() for t in figure.axes[0].texts]
        assert any("No output bounds present" in t for t in texts)
    finally:
        plt.close(figure)


def test_cohort_visualiser_handles_all_nan_gracefully() -> None:
    n, k = 3, 4
    nan = torch.full((n, k), float("nan"))
    result = _formal_result(n=n, k=k, output_bounds={"lower": nan.clone(), "upper": nan.clone()})
    figure = OutputBoundsCohortVisualiser().visualise(result, context=_ctx())
    try:
        texts = [t.get_text() for t in figure.axes[0].texts]
        assert any("No output bounds present" in t for t in texts)
    finally:
        plt.close(figure)


def test_cohort_method_kind_rejection_for_empirical_results() -> None:
    visualiser = OutputBoundsCohortVisualiser()
    with pytest.raises(MethodKindVisualiserIncompatibilityError):
        visualiser.validate_result(_empirical_result())


# ---------------------------- pinned ---------------------------------------


def test_pinned_visualiser_default_picks_first_finite_samples() -> None:
    n, k = 5, 4
    lower = torch.full((n, k), float("nan"))
    upper = torch.full((n, k), float("nan"))
    for j in range(k):
        lower[0, j] = -1.0
        upper[0, j] = 1.0
        lower[2, j] = -0.5
        upper[2, j] = 0.5
    result = _formal_result(n=n, k=k, output_bounds={"lower": lower, "upper": upper})
    figure = OutputBoundsPinnedVisualiser(max_samples=4).visualise(result, context=_ctx())
    try:
        # 2 finite samples → 2 subplots.
        assert len(figure.axes) == 2
        titles = [ax.get_title() for ax in figure.axes]
        assert any("sample 0" in t for t in titles)
        assert any("sample 2" in t for t in titles)
    finally:
        plt.close(figure)


def test_pinned_visualiser_uses_sample_indices_kwarg_when_present() -> None:
    result = _formal_result(n=5, k=4)
    visualiser = OutputBoundsPinnedVisualiser(sample_indices=[3])
    figure = visualiser.visualise(result, context=_ctx())
    try:
        assert len(figure.axes) == 1
        assert "sample 3" in figure.axes[0].get_title()
    finally:
        plt.close(figure)


def test_pinned_visualiser_highlights_target_class() -> None:
    n, k = 1, 4
    lower = torch.full((n, k), -1.0)
    upper = torch.full((n, k), 1.0)
    result = _formal_result(n=n, k=k, output_bounds={"lower": lower, "upper": upper})
    # target for sample 0 is 0 % 4 == 0
    visualiser = OutputBoundsPinnedVisualiser(target_color="#d62728", bar_color="#1f77b4")
    figure = visualiser.visualise(result, context=_ctx())
    try:
        ax = figure.axes[0]
        # The hlines collections store their colors as RGBA arrays.
        colors_seen: set[tuple[float, float, float, float]] = set()
        for coll in ax.collections:
            get_colors = getattr(coll, "get_colors", None)
            if get_colors is None:
                continue
            for rgba in get_colors():
                colors_seen.add(tuple(rgba))
        from matplotlib.colors import to_rgba

        assert to_rgba("#d62728") in colors_seen
        assert to_rgba("#1f77b4") in colors_seen
    finally:
        plt.close(figure)


def test_pinned_visualiser_handles_none_bounds_gracefully() -> None:
    result = _formal_result(output_bounds=None)
    figure = OutputBoundsPinnedVisualiser().visualise(result, context=_ctx())
    try:
        texts = [t.get_text() for t in figure.axes[0].texts]
        assert any("No output bounds present" in t for t in texts)
    finally:
        plt.close(figure)


def test_pinned_method_kind_rejection_for_empirical_results() -> None:
    visualiser = OutputBoundsPinnedVisualiser()
    with pytest.raises(MethodKindVisualiserIncompatibilityError):
        visualiser.validate_result(_empirical_result())


# ---------------------------- width heatmap (C1) ---------------------------


def test_width_heatmap_renders_grid_with_colorbar() -> None:
    visualiser = OutputBoundsWidthHeatmapVisualiser()
    result = _formal_result(n=5, k=4)
    figure = visualiser.visualise(result, context=_ctx())
    try:
        from matplotlib.image import AxesImage

        images = [im for ax in figure.axes for im in ax.get_images()]
        assert len(images) == 1
        assert isinstance(images[0], AxesImage)
        # The figure has 2 axes when a colorbar is attached (image + colorbar).
        assert len(figure.axes) == 2
        main_ax = images[0].axes
        assert len(main_ax.get_xticklabels()) == 4
        labels = [t.get_text() for t in main_ax.get_xticklabels()]
        assert labels == [f"logit_{k}" for k in range(4)]
    finally:
        plt.close(figure)


def test_width_heatmap_masks_all_nan_row() -> None:
    n, k = 4, 3
    lower = torch.zeros(n, k)
    upper = torch.ones(n, k)
    lower[2] = float("nan")
    upper[2] = float("nan")
    result = _formal_result(n=n, k=k, output_bounds={"lower": lower, "upper": upper})
    figure = OutputBoundsWidthHeatmapVisualiser().visualise(result, context=_ctx())
    try:
        import numpy as np

        images = [im for ax in figure.axes for im in ax.get_images()]
        arr = images[0].get_array()
        assert hasattr(arr, "mask")
        mask = np.asarray(arr.mask)
        # Row 2 must be fully masked.
        assert mask[2].all()
        # Rows 0,1,3 must be fully unmasked.
        assert not mask[0].any()
        assert not mask[1].any()
        assert not mask[3].any()
    finally:
        plt.close(figure)


def test_width_heatmap_handles_none_bounds_gracefully() -> None:
    result = _formal_result(output_bounds=None)
    figure = OutputBoundsWidthHeatmapVisualiser().visualise(result, context=_ctx())
    try:
        texts = [t.get_text() for t in figure.axes[0].texts]
        assert any("No output bounds present" in t for t in texts)
    finally:
        plt.close(figure)


def test_width_heatmap_method_kind_rejection_for_empirical_results() -> None:
    with pytest.raises(MethodKindVisualiserIncompatibilityError):
        OutputBoundsWidthHeatmapVisualiser().validate_result(_empirical_result())


# ---------------------------- margin heatmap (C2) --------------------------


def test_margin_heatmap_centers_on_zero() -> None:
    # Construct bounds where margin spans both signs:
    # target column lower is moderate; some non-target uppers above it (negative margin)
    # and some below (positive margin).
    n, k = 3, 4
    lower = torch.zeros(n, k)
    upper = torch.zeros(n, k)
    # targets are 0,1,2 (n % k)
    # sample 0, target=0: lower[0,0]=2.0; upper[0,1]=1.0 (margin=+1), upper[0,2]=3.0 (margin=-1)
    lower[0, 0] = 2.0
    upper[0, 0] = 5.0
    upper[0, 1] = 1.0
    upper[0, 2] = 3.0
    upper[0, 3] = 0.0
    lower[1, 1] = 0.0
    upper[1, 1] = 5.0
    upper[1, 0] = -1.0
    upper[1, 2] = 2.0
    upper[1, 3] = 4.0
    lower[2, 2] = 1.0
    upper[2, 2] = 5.0
    upper[2, 0] = 0.5
    upper[2, 1] = 3.0
    upper[2, 3] = 2.5
    result = _formal_result(n=n, k=k, output_bounds={"lower": lower, "upper": upper})
    figure = OutputBoundsMarginHeatmapVisualiser().visualise(result, context=_ctx())
    try:
        from matplotlib.colors import TwoSlopeNorm

        images = [im for ax in figure.axes for im in ax.get_images()]
        assert len(images) == 1
        norm = images[0].norm
        assert isinstance(norm, TwoSlopeNorm)
        assert norm.vcenter == 0.0
    finally:
        plt.close(figure)


def test_margin_heatmap_masks_target_column() -> None:
    n, k = 5, 4
    lower = torch.zeros(n, k)
    upper = torch.ones(n, k)
    result = _formal_result(n=n, k=k, output_bounds={"lower": lower, "upper": upper})
    figure = OutputBoundsMarginHeatmapVisualiser().visualise(result, context=_ctx())
    try:
        import numpy as np

        images = [im for ax in figure.axes for im in ax.get_images()]
        arr = images[0].get_array()
        mask = np.asarray(arr.mask)
        targets = result.targets.numpy()
        for i in range(n):
            assert mask[i, int(targets[i])], f"target cell at row {i} not masked"
    finally:
        plt.close(figure)


def test_margin_heatmap_falls_back_when_targets_missing() -> None:
    result = _formal_result(n=3, k=4)
    object.__setattr__(result, "targets", None)
    figure = OutputBoundsMarginHeatmapVisualiser().visualise(result, context=_ctx())
    try:
        texts = [t.get_text() for t in figure.axes[0].texts]
        assert any("requires per-sample targets" in t for t in texts)
    finally:
        plt.close(figure)


def test_margin_heatmap_method_kind_rejection_for_empirical_results() -> None:
    with pytest.raises(MethodKindVisualiserIncompatibilityError):
        OutputBoundsMarginHeatmapVisualiser().validate_result(_empirical_result())
