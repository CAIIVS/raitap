"""Regenerate documentation preview PNGs for every robustness visualiser.

Run this whenever a visualiser's appearance changes:

    MPLBACKEND=Agg uv run python scripts/generate_visualiser_previews.py

Outputs land in ``docs/_static/visualisers/`` and are committed alongside the
docs page that references them.
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
    MethodKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessSemantics,
    RobustnessVerdict,
    RobustnessVisualisationContext,
    ThreatModel,
)
from raitap.robustness.results import RobustnessMetrics, RobustnessResult, encode_verdicts
from raitap.robustness.visualisers import (
    ImagePairVisualiser,
    OutputBoundsCohortVisualiser,
    OutputBoundsMarginHeatmapVisualiser,
    OutputBoundsPinnedVisualiser,
    OutputBoundsWidthHeatmapVisualiser,
    PerturbationHeatmapVisualiser,
    VerdictSummaryVisualiser,
)
from raitap.transparency.contracts import InputKind, InputSpec

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
            method_kind=MethodKind.FORMAL_VERIFICATION,
            threat_model=ThreatModel.WHITE_BOX,
            objective=Objective.UNTARGETED,
            families=frozenset({"smt"}),
            budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
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
        RobustnessVerdict.ATTACKED,
        RobustnessVerdict.ATTACKED,
        RobustnessVerdict.NOT_ATTACKED,
        RobustnessVerdict.ATTACKED,
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
            method_kind=MethodKind.EMPIRICAL_ATTACK,
            threat_model=ThreatModel.WHITE_BOX,
            objective=Objective.UNTARGETED,
            families=frozenset({"gradient"}),
            budget=PerturbationBudget(norm=PerturbationNorm.LINF, epsilon=0.05),
            input_spec=InputSpec(kind=InputKind.IMAGE, shape=(n, c, h, w), layout="NCHW"),
        ),
    )


def _ctx(method_kind: MethodKind, algorithm: str) -> RobustnessVisualisationContext:
    return RobustnessVisualisationContext(
        algorithm=algorithm,
        method_kind=method_kind,
        sample_names=None,
        show_sample_names=False,
    )


def _save(name: str, fig: Figure) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    empirical = _empirical_fixture()
    formal = _formal_fixture()

    empirical_ctx = _ctx(MethodKind.EMPIRICAL_ATTACK, "PGD")
    formal_ctx = _ctx(MethodKind.FORMAL_VERIFICATION, "marabou-linf")

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

    for p in saved:
        print(f"wrote {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
