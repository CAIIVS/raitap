"""Regression test for issue #197: assess_robustness short-circuits before
touching ``data.tensor`` when ``task_kind == detection``.

For detection, ``data.tensor`` is a ragged ``list[torch.Tensor]`` (variable
per-image sizes).  The classification path later indexes it as a dense
``(N, C, H, W)`` tensor via ``RobustnessAssessment``.  The guard at line ~78
of ``assess_robustness.py`` must fire *before* that code path is reached so
that no ``AttributeError`` / ``TypeError`` is raised on the ragged list.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from raitap.configs.schema import AppConfig, RobustnessConfig
from raitap.pipeline.outputs import ForwardOutput
from raitap.pipeline.phases.assess_robustness import assess_robustness
from raitap.types import TaskKind


def _make_detection_forward_output(num_samples: int = 2) -> ForwardOutput:
    """Minimal detection ForwardOutput that satisfies __post_init__."""
    return ForwardOutput(
        task_kind=TaskKind.detection,
        batch_size=num_samples,
        detection_predictions=[
            {
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.int64),
            }
            for _ in range(num_samples)
        ],
    )


class _FakeData:
    """Data stub with a ragged list[Tensor] as used by detection pipelines."""

    def __init__(self, num_samples: int = 2) -> None:
        # Deliberately ragged — different spatial sizes per image.
        self.tensor: list[torch.Tensor] = [
            torch.zeros(3, 40, 50),
            torch.zeros(3, 60, 30),
        ]
        self.sample_ids = [f"img_{i}.jpg" for i in range(num_samples)]


def test_assess_robustness_detection_returns_empty_and_skips_robustness_assessment() -> None:
    """assess_robustness returns ([], []) for detection without touching data.tensor.

    Regression for #197: when data.tensor is a ragged list[Tensor] the
    detection guard must fire *before* RobustnessAssessment is constructed
    (which would index data.tensor as a dense (N,C,H,W) tensor).
    """
    config = AppConfig(experiment_name="regression-197")
    # Populate robustness so the first ``if not assessors`` guard does NOT
    # short-circuit — we need to reach the task-kind check.
    config.robustness = {
        "fgsm": RobustnessConfig(
            _target_="EmpiricalAttackAssessor",
            algorithm="FGSM",
        )
    }

    data = _FakeData(num_samples=2)
    forward = _make_detection_forward_output(num_samples=2)

    # Belt + suspenders: if the task-kind guard is ever removed or reordered,
    # RobustnessAssessment would be called — catch that loudly.
    with patch(
        "raitap.pipeline.phases.assess_robustness.RobustnessAssessment",
        side_effect=AssertionError("RobustnessAssessment must not be reached for detection"),
    ):
        results = assess_robustness(
            config,
            model=None,  # type: ignore[arg-type]
            data=data,  # type: ignore[arg-type]
            forward_output=forward,
            labels=None,
            input_metadata=None,
            resolved_preprocessing=None,
        )

    assert results == [], "expected empty results for detection task"
