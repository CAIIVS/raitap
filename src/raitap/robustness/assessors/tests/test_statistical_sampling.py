"""Aggregation tests for StatisticalSamplingAssessor using a stub perturbation.

A stub (identity) apply_perturbation lets us prove the accuracy/verdict/CI
aggregation without the real imagecorruptions dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

from raitap.robustness.assessors.base_assessor import StatisticalSamplingAssessor
from raitap.robustness.contracts import (
    AssessmentKind,
    Objective,
    RobustnessVerdict,
    ThreatModel,
)
from raitap.robustness.results import decode_verdicts
from raitap.robustness.semantics import AssessorAlgorithmSpec


class _OneHotModel(nn.Module):
    """Predicts class == round(mean of image * 2) clipped to [0,2] — deterministic."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        logits = torch.zeros(n, 3)
        for i in range(n):
            cls = int(min(2, max(0, round(float(x[i].mean()) * 2))))
            logits[i, cls] = 1.0
        return logits


class _StubSamplingAssessor(StatisticalSamplingAssessor):
    algorithm_registry: ClassVar[Mapping[str, AssessorAlgorithmSpec]] = {
        "identity": AssessorAlgorithmSpec(
            AssessmentKind.STATISTICAL_SAMPLING,
            ThreatModel.NOT_APPLICABLE,
            Objective.UNTARGETED,
            families={"stub"},
        )
    }

    def __init__(self, algorithm: str = "identity", *, severity: int = 1, **kw: Any) -> None:
        self.algorithm = algorithm
        self.severity = severity
        self.init_kwargs = dict(kw)

    def apply_perturbation(self, image: np.ndarray) -> np.ndarray:
        return image  # identity stub


def test_aggregation_counts_correct_and_builds_ci(tmp_path: Any) -> None:
    assessor = _StubSamplingAssessor()
    inputs = torch.stack(
        [
            torch.full((3, 4, 4), 0.0),  # mean 0 -> class 0
            torch.full((3, 4, 4), 0.5),  # mean .5 -> class 1
            torch.full((3, 4, 4), 1.0),  # mean 1 -> class 2
        ]
    )
    targets = torch.tensor([0, 1, 0])  # third deliberately wrong (pred 2 != 0)

    result = assessor.assess(_OneHotModel(), inputs, targets, run_dir=str(tmp_path))

    assert result.metrics.n_samples == 3
    assert result.metrics.n_correct == 2
    assert result.metrics.corrupted_accuracy == 2 / 3
    ci_low = result.metrics.accuracy_ci_low
    ci_high = result.metrics.accuracy_ci_high
    accuracy = result.metrics.corrupted_accuracy
    assert ci_low is not None and ci_high is not None and accuracy is not None
    assert 0.0 <= ci_low <= accuracy <= ci_high <= 1.0

    verdicts = decode_verdicts(result.verdicts)
    assert verdicts == [
        RobustnessVerdict.CORRECT_UNDER_PERTURBATION,
        RobustnessVerdict.CORRECT_UNDER_PERTURBATION,
        RobustnessVerdict.MISCLASSIFIED_UNDER_PERTURBATION,
    ]
    assert result.assessment_kind is AssessmentKind.STATISTICAL_SAMPLING
