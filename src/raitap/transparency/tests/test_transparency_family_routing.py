"""Routing test: ``assess_transparency`` resolves a TaskFamily and delegates.

The phase no longer branches on ``task_kind``. Instead it resolves the
``TaskFamily`` for the forward output's kind, runs the shared
``prepare_explainer`` setup once per explainer, and calls ``family.explain``.
These tests pin that contract without booting a real backend or explainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from raitap.configs import set_output_root
from raitap.configs.schema import AppConfig, TransparencyConfig
from raitap.task_families.registry import resolve_task_family
from raitap.types import TaskKind

from .test_assess_transparency_detection import (
    _FakeData,
    _FakeModel,
    _make_detection_forward_output,
)


def test_each_family_exposes_explain() -> None:
    for kind in (TaskKind.classification, TaskKind.detection):
        assert hasattr(resolve_task_family(kind), "explain")


def test_assess_transparency_resolves_family_and_calls_explain(tmp_path: Path) -> None:
    from raitap.transparency import phase as phase_mod

    config = AppConfig(experiment_name="routing-family-test")
    set_output_root(config, tmp_path)
    config.transparency = {
        "ig_det": TransparencyConfig(
            use="captum",
            algorithm="IntegratedGradients",
            call={"target": 0},
            raitap={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            visualisers=[{"use": "detection_image"}],
        )
    }

    model = _FakeModel()
    data = _FakeData(num_samples=2)
    forward = _make_detection_forward_output(num_samples=2)

    fake_family = MagicMock()
    fake_family.explain.return_value = []
    with (
        patch.object(phase_mod, "resolve_task_family", return_value=fake_family) as resolve,
        patch.object(phase_mod, "prepare_explainer", return_value=object()),
    ):
        phase_mod.assess_transparency(
            config,
            model,  # type: ignore[arg-type]
            data,  # type: ignore[arg-type]
            forward,
            input_metadata=None,
            resolved_preprocessing=None,
        )
        resolve.assert_called()
        fake_family.explain.assert_called()
