from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

from raitap.robustness.contracts import (
    MethodKind,
    Objective,
    PerturbationBudget,
    PerturbationNorm,
    RobustnessSemantics,
    RobustnessVerdict,
    ThreatModel,
)
from raitap.robustness.results import (
    RobustnessMetrics,
    RobustnessResult,
    encode_verdicts,
)

if TYPE_CHECKING:
    from pathlib import Path


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
