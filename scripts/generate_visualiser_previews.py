"""Regenerate documentation preview PNGs for every robustness + transparency visualiser.

Run this whenever a visualiser's appearance changes:

    MPLBACKEND=Agg uv run python scripts/generate_visualiser_previews.py

Outputs land in ``docs/_static/visualisers/`` and are committed alongside the
docs pages that reference them.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from raitap.robustness.contracts import (
    AssessmentKind,
    Objective,
    PerturbationBudget,
    PerturbationDistribution,
    PerturbationNorm,
    RobustnessSemantics,
    RobustnessVerdict,
    RobustnessVisualisationContext,
    ThreatModel,
)
from raitap.robustness.results import RobustnessMetrics, RobustnessResult, encode_verdicts
from raitap.robustness.visualisers import (
    CorruptionAccuracyVisualiser,
    ImagePairVisualiser,
    OutputBoundsCohortVisualiser,
    OutputBoundsMarginHeatmapVisualiser,
    OutputBoundsPinnedVisualiser,
    OutputBoundsWidthHeatmapVisualiser,
    PerturbationHeatmapVisualiser,
    VerdictSummaryVisualiser,
)
from raitap.transparency.contracts import (
    DetectionBox,
    InputKind,
    InputSpec,
    StructuredPayload,
    StructuredPayloadKind,
)
from raitap.transparency.contracts import (
    VisualisationContext as TransparencyVisualisationContext,
)
from raitap.transparency.visualisers import (
    CaptumImageVisualiser,
    CaptumTextVisualiser,
    CaptumTimeSeriesVisualiser,
    DetectionImageVisualiser,
    InputThumbnailVisualiser,
    ShapBarVisualiser,
    ShapBeeswarmVisualiser,
    ShapForceVisualiser,
    ShapImageVisualiser,
    ShapWaterfallVisualiser,
    StructuredPayloadSummaryVisualiser,
    TabularBarChartVisualiser,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "_static" / "visualisers"


def _formal_fixture() -> RobustnessResult:
    """ACAS-Xu-shaped formal result with interesting bounds + a NaN row."""
    rng = np.random.default_rng(0)
    n, k = 8, 5
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    # Build per-sample bounds so widths vary across classes (interesting heatmap).
    lower = torch.zeros(n, k)
    upper = torch.zeros(n, k)
    for i in range(n):
        target = int(targets[i])
        base_lower = float(rng.uniform(0.5, 1.5))
        base_upper = base_lower + float(rng.uniform(0.4, 1.6))
        for j in range(k):
            if j == target:
                lower[i, j] = base_upper + rng.uniform(0.1, 0.6)
                upper[i, j] = lower[i, j] + rng.uniform(0.2, 0.8)
            else:
                lower[i, j] = base_lower - rng.uniform(0.0, 0.6)
                # vary upper to create both robust and borderline samples
                offset = rng.uniform(0.0, 2.0)
                if i in (1, 5) and j == (target + 1) % k:
                    # one "borderline" non-target overtakes target upper bound
                    upper[i, j] = lower[i, target].item() + 0.2
                else:
                    upper[i, j] = lower[i, j] + offset
    # One row with no certified bounds (UNKNOWN / FALSIFIED case).
    lower[6] = float("nan")
    upper[6] = float("nan")

    verdicts = [RobustnessVerdict.VERIFIED] * n
    verdicts[6] = RobustnessVerdict.UNKNOWN
    verdicts[1] = RobustnessVerdict.FALSIFIED

    runtimes = torch.tensor(rng.uniform(0.05, 2.5, size=n), dtype=torch.float32)
    runtimes[6] = 5.0  # the unknown case is the long-tail timeout

    return RobustnessResult(
        clean_inputs=torch.zeros(n, 5),
        targets=targets,
        clean_predictions=targets.clone(),
        verdicts=encode_verdicts(verdicts),
        metrics=RobustnessMetrics(
            clean_accuracy=1.0,
            verified_rate=6 / 8,
            falsified_rate=1 / 8,
            unknown_rate=1 / 8,
            mean_runtime=float(runtimes.mean()),
        ),
        run_dir=Path("."),
        experiment_name="marabou-preview",
        assessor_target="raitap.robustness.assessors.MarabouAssessor",
        algorithm="marabou-linf",
        assessor_name="marabou_linf",
        output_bounds={"lower": lower, "upper": upper},
        runtime_per_sample=runtimes,
        semantics=RobustnessSemantics(
            assessment_kind=AssessmentKind.FORMAL_VERIFICATION,
            threat_model=ThreatModel.WHITE_BOX,
            objective=Objective.UNTARGETED,
            families=frozenset({"smt"}),
            perturbation=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
        ),
    )


def _empirical_fixture() -> RobustnessResult:
    """Tiny RGB image batch + perturbations so the empirical visualisers light up."""
    rng = np.random.default_rng(1)
    n, c, h, w = 4, 3, 16, 16
    clean = torch.tensor(rng.uniform(0.0, 1.0, size=(n, c, h, w)), dtype=torch.float32)
    perturbation = torch.tensor(rng.normal(0.0, 0.08, size=(n, c, h, w)), dtype=torch.float32)
    perturbed = (clean + perturbation).clamp(0.0, 1.0)
    targets = torch.tensor([0, 1, 2, 3])
    verdicts = [
        RobustnessVerdict.ATTACK_SUCCEEDED,
        RobustnessVerdict.ATTACK_SUCCEEDED,
        RobustnessVerdict.ATTACK_FAILED,
        RobustnessVerdict.ATTACK_SUCCEEDED,
    ]
    distance = torch.tensor([0.05, 0.04, float("nan"), 0.06])
    perturbed_predictions = torch.tensor([1, 0, 2, 0])

    return RobustnessResult(
        clean_inputs=clean,
        targets=targets,
        clean_predictions=targets.clone(),
        perturbed_inputs=perturbed,
        perturbed_predictions=perturbed_predictions,
        perturbation_distance=distance,
        verdicts=encode_verdicts(verdicts),
        metrics=RobustnessMetrics(
            clean_accuracy=1.0,
            adversarial_accuracy=0.25,
            attack_success_rate=0.75,
            mean_distance=0.05,
            max_distance=0.06,
        ),
        run_dir=Path("."),
        experiment_name="pgd-preview",
        assessor_target="raitap.robustness.assessors.TorchattacksAssessor",
        algorithm="PGD",
        assessor_name="pgd",
        semantics=RobustnessSemantics(
            assessment_kind=AssessmentKind.EMPIRICAL_ATTACK,
            threat_model=ThreatModel.WHITE_BOX,
            objective=Objective.UNTARGETED,
            families=frozenset({"gradient"}),
            perturbation=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
            input_spec=InputSpec(kind=InputKind.IMAGE, shape=(n, c, h, w), layout="NCHW"),
        ),
    )


def _sampling_fixture() -> RobustnessResult:
    """Statistical-sampling result so the average-case visualiser lights up."""
    n = 24
    n_correct = 15
    verdicts = [RobustnessVerdict.CORRECT_UNDER_PERTURBATION] * n_correct + [
        RobustnessVerdict.MISCLASSIFIED_UNDER_PERTURBATION
    ] * (n - n_correct)
    return RobustnessResult(
        clean_inputs=torch.zeros(n, 3, 16, 16),
        targets=torch.zeros(n, dtype=torch.long),
        clean_predictions=torch.zeros(n, dtype=torch.long),
        verdicts=encode_verdicts(verdicts),
        metrics=RobustnessMetrics(
            clean_accuracy=0.83,
            corrupted_accuracy=n_correct / n,
            accuracy_ci_low=0.43,
            accuracy_ci_high=0.80,
            n_samples=n,
            n_correct=n_correct,
        ),
        run_dir=Path("."),
        experiment_name="imagecorruptions-preview",
        assessor_target="raitap.robustness.assessors.ImageCorruptionsAssessor",
        algorithm="gaussian_noise",
        assessor_name="gaussian_noise",
        semantics=RobustnessSemantics(
            assessment_kind=AssessmentKind.STATISTICAL_SAMPLING,
            threat_model=ThreatModel.NOT_APPLICABLE,
            objective=Objective.UNTARGETED,
            families=frozenset({"common_corruption", "noise"}),
            perturbation=PerturbationDistribution(corruption_name="gaussian_noise", severity=3),
        ),
    )


def _ctx(assessment_kind: AssessmentKind, algorithm: str) -> RobustnessVisualisationContext:
    return RobustnessVisualisationContext(
        algorithm=algorithm,
        assessment_kind=assessment_kind,
        sample_names=None,
        show_sample_names=False,
    )


def _save(name: str, fig: Figure) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _transparency_image_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    """RGB image batch + attribution tensor with a centred salient patch."""
    rng = np.random.default_rng(2)
    n, c, h, w = 2, 3, 32, 32
    image = torch.tensor(rng.uniform(0.1, 0.9, size=(n, c, h, w)), dtype=torch.float32)
    attributions = torch.tensor(rng.normal(0.0, 0.05, size=(n, c, h, w)), dtype=torch.float32)
    # Add a strong salient blob in the centre so the heatmap has structure.
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    blob = np.exp(-(((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 5.0**2)))
    for i in range(n):
        attributions[i, 0] += torch.tensor(blob, dtype=torch.float32) * (0.8 if i == 0 else -0.6)
    return image, attributions


def _transparency_time_series_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    """(T, C) signal + attribution overlay — Captum convention is channels-last."""
    t, c = 50, 2
    t_axis = np.linspace(0, 4 * np.pi, t)
    signal = np.stack([np.sin(t_axis), 0.5 * np.cos(2 * t_axis)], axis=-1)
    rng = np.random.default_rng(3)
    attributions = rng.normal(0.0, 0.05, size=(t, c))
    # Mark the second half of channel 0 as "important".
    attributions[t // 2 :, 0] += np.linspace(0.0, 0.4, t - t // 2)
    return (
        torch.tensor(signal, dtype=torch.float32),
        torch.tensor(attributions, dtype=torch.float32),
    )


def _transparency_text_fixture() -> tuple[list[str], torch.Tensor]:
    """Short tokenised sentence + per-token attributions."""
    tokens = ["The", "model", "really", "disliked", "the", "predicted", "outcome", "."]
    rng = np.random.default_rng(4)
    attributions = torch.tensor(rng.uniform(-0.4, 0.6, size=len(tokens)), dtype=torch.float32)
    # Make the "disliked" token strongly negative for an interpretable picture.
    attributions[3] = -0.85
    attributions[2] = 0.7
    return tokens, attributions


def _transparency_tabular_fixture() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """(N, F) feature matrix + SHAP-shaped (N, F) attributions + feature names."""
    rng = np.random.default_rng(5)
    n, f = 100, 5
    feature_names = ["age", "income", "credit_score", "tenure", "balance"]
    features = torch.tensor(rng.normal(0.0, 1.0, size=(n, f)), dtype=torch.float32)
    # SHAP-style attributions correlated with feature value for visual interest.
    attributions_np = rng.normal(0.0, 0.3, size=(n, f))
    attributions_np[:, 0] += 0.5 * features[:, 0].numpy()
    attributions_np[:, 1] -= 0.4 * features[:, 1].numpy()
    attributions_np[:, 2] += 0.6 * features[:, 2].numpy()
    return features, torch.tensor(attributions_np, dtype=torch.float32), feature_names


def _shap_unavailable_placeholder(name: str, error: Exception) -> Figure:
    """Render a tombstone PNG so the docs link still resolves when SHAP fails."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(
        0.5,
        0.6,
        f"{name} preview unavailable in this environment",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )
    ax.text(
        0.5,
        0.35,
        "Regenerate on a host with `shap` installed.",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )
    ax.text(
        0.5,
        0.15,
        f"({type(error).__name__}: {error})",
        ha="center",
        va="center",
        fontsize=7,
        color="grey",
    )
    return fig


def _render_transparency_previews() -> list[Path]:
    saved: list[Path] = []
    image_inputs, image_attrs = _transparency_image_fixture()
    ts_inputs, ts_attrs = _transparency_time_series_fixture()
    tokens, text_attrs = _transparency_text_fixture()
    tabular_inputs, tabular_attrs, feature_names = _transparency_tabular_fixture()

    captum_ctx = TransparencyVisualisationContext(
        algorithm="IntegratedGradients",
        sample_names=None,
        show_sample_names=False,
    )
    shap_ctx = TransparencyVisualisationContext(
        algorithm="GradientExplainer",
        sample_names=None,
        show_sample_names=False,
    )

    # Captum
    saved.append(
        _save(
            "captum_image_visualiser",
            CaptumImageVisualiser(method="blended_heat_map", sign="all").visualise(
                image_attrs, inputs=image_inputs, context=captum_ctx, max_samples=2
            ),
        )
    )
    ts_viz = CaptumTimeSeriesVisualiser(method="overlay_individual", sign="absolute_value")
    saved.append(
        _save(
            "captum_time_series_visualiser",
            ts_viz.visualise(ts_attrs, inputs=ts_inputs, context=captum_ctx),
        )
    )
    saved.append(
        _save(
            "captum_text_visualiser",
            CaptumTextVisualiser().visualise(text_attrs, token_labels=tokens),
        )
    )

    # SHAP — each call gated so a runtime breakage still emits a usable PNG.
    def _shap_save(name: str, builder: callable) -> None:  # type: ignore[valid-type]
        try:
            fig = builder()
        except Exception as exc:
            print(f"warning: {name} failed ({exc!r}); writing placeholder")
            fig = _shap_unavailable_placeholder(name, exc)
        saved.append(_save(name, fig))

    _shap_save(
        "shap_bar_visualiser",
        lambda: ShapBarVisualiser(feature_names=feature_names).visualise(
            tabular_attrs, inputs=tabular_inputs, context=shap_ctx
        ),
    )
    _shap_save(
        "shap_beeswarm_visualiser",
        lambda: ShapBeeswarmVisualiser(feature_names=feature_names).visualise(
            tabular_attrs, inputs=tabular_inputs, context=shap_ctx
        ),
    )
    _shap_save(
        "shap_waterfall_visualiser",
        lambda: ShapWaterfallVisualiser(
            feature_names=feature_names, expected_value=0.0, sample_index=0
        ).visualise(tabular_attrs, inputs=tabular_inputs, context=shap_ctx),
    )
    _shap_save(
        "shap_force_visualiser",
        lambda: ShapForceVisualiser(
            feature_names=feature_names, expected_value=0.0, sample_index=0
        ).visualise(tabular_attrs, inputs=tabular_inputs, context=shap_ctx),
    )
    _shap_save(
        "shap_image_visualiser",
        lambda: ShapImageVisualiser(max_samples=2).visualise(
            image_attrs, inputs=image_inputs, context=shap_ctx
        ),
    )

    # Generic
    saved.append(
        _save(
            "tabular_bar_chart_visualiser",
            TabularBarChartVisualiser(feature_names=feature_names).visualise(
                tabular_attrs, context=captum_ctx
            ),
        )
    )

    # Structured payload summary (per-sample convergence-delta diagnostics from an
    # IntegratedGradients run with return_convergence_delta=True).
    structured_ctx = TransparencyVisualisationContext(
        algorithm="IntegratedGradients",
        sample_names=None,
        show_sample_names=False,
        structured_payloads=(
            StructuredPayload(
                "convergence_delta",
                StructuredPayloadKind.CONVERGENCE_DELTA,
                torch.tensor([0.004, -0.021, 0.038, 0.009, -0.006, 0.045, 0.012, -0.030]),
            ),
        ),
    )
    saved.append(
        _save(
            "structured_payload_summary_visualiser",
            StructuredPayloadSummaryVisualiser().visualise(
                torch.zeros(8, 4), context=structured_ctx
            ),
        )
    )

    # Input thumbnail (reporting sample-header preview)
    saved.append(
        _save(
            "input_thumbnail_visualiser",
            InputThumbnailVisualiser().visualise(
                image_attrs, inputs=image_inputs, context=captum_ctx, max_samples=2
            ),
        )
    )

    # Detection (one figure per box; needs detection_box on the context)
    detection_ctx = TransparencyVisualisationContext(
        algorithm="IntegratedGradients",
        sample_names=None,
        show_sample_names=False,
        detection_box=DetectionBox(
            display_index=0,
            raw_index=3,
            xyxy=(6.0, 5.0, 26.0, 24.0),
            score=0.92,
            label_index=17,
            label_name="cat",
        ),
    )
    saved.append(
        _save(
            "detection_image_visualiser",
            DetectionImageVisualiser().visualise(
                image_attrs[0], inputs=image_inputs[0], context=detection_ctx
            ),
        )
    )
    return saved


def main() -> None:
    empirical = _empirical_fixture()
    formal = _formal_fixture()

    empirical_ctx = _ctx(AssessmentKind.EMPIRICAL_ATTACK, "PGD")
    formal_ctx = _ctx(AssessmentKind.FORMAL_VERIFICATION, "marabou-linf")

    saved: list[Path] = []
    saved.append(
        _save(
            "image_pair_visualiser",
            ImagePairVisualiser(max_samples=3).visualise(empirical, context=empirical_ctx),
        )
    )
    saved.append(
        _save(
            "perturbation_heatmap_visualiser",
            PerturbationHeatmapVisualiser(max_samples=3).visualise(
                empirical, context=empirical_ctx
            ),
        )
    )
    saved.append(
        _save(
            "verdict_summary_visualiser",
            VerdictSummaryVisualiser().visualise(formal, context=formal_ctx),
        )
    )
    saved.append(
        _save(
            "output_bounds_cohort_visualiser",
            OutputBoundsCohortVisualiser().visualise(formal, context=formal_ctx),
        )
    )
    saved.append(
        _save(
            "output_bounds_pinned_visualiser",
            OutputBoundsPinnedVisualiser(max_samples=3).visualise(formal, context=formal_ctx),
        )
    )
    saved.append(
        _save(
            "output_bounds_width_heatmap_visualiser",
            OutputBoundsWidthHeatmapVisualiser().visualise(formal, context=formal_ctx),
        )
    )
    saved.append(
        _save(
            "output_bounds_margin_heatmap_visualiser",
            OutputBoundsMarginHeatmapVisualiser().visualise(formal, context=formal_ctx),
        )
    )

    sampling = _sampling_fixture()
    sampling_ctx = _ctx(AssessmentKind.STATISTICAL_SAMPLING, "gaussian_noise")
    saved.append(
        _save(
            "corruption_accuracy_visualiser",
            CorruptionAccuracyVisualiser().visualise(sampling, context=sampling_ctx),
        )
    )

    saved.extend(_render_transparency_previews())

    for p in saved:
        print(f"wrote {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
