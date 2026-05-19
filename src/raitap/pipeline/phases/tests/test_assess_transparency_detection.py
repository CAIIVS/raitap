"""Routing test for ``assess_transparency`` when ``task_kind == detection``.

The detection branch (``_assess_transparency_detection``) duplicates the
per-explainer setup that ``Explanation.__call__`` does for classification
(create explainer + visualisers, compat checks, resolve backend +
run_dir). This test asserts the routing reaches ``explain_detection`` with
the right kwargs and emits one ExplanationResult per kept detection box —
without booting a real backend or explainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from raitap.transparency.results import ExplanationResult

from raitap.configs import set_output_root
from raitap.configs.schema import AppConfig, TransparencyConfig
from raitap.models.backend import ModelBackend
from raitap.pipeline.outputs import ForwardOutput
from raitap.pipeline.phases.assess_transparency import assess_transparency
from raitap.types import TaskKind


class _FakeBackend(ModelBackend):
    """Minimal backend stub the detection branch only needs to read.

    Subclasses :class:`ModelBackend` so ``_require_model_backend`` accepts it.
    """

    supports_torch_autograd = True

    def __init__(self) -> None:
        self._task_kind = TaskKind.detection

    @property
    def task_kind(self) -> TaskKind:
        return self._task_kind

    @property
    def hardware_label(self) -> str:
        return "fake"

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs

    def __call__(self, inputs: torch.Tensor) -> Any:
        raise NotImplementedError("routing test mocks the explain path")

    def as_model_for_explanation(self) -> Any:
        raise NotImplementedError("routing test mocks the explain path")


class _FakeModel:
    def __init__(self) -> None:
        self.backend = _FakeBackend()


def _make_detection_forward_output(num_samples: int) -> ForwardOutput:
    return ForwardOutput(
        task_kind=TaskKind.detection,
        batch_size=num_samples,
        detection_predictions=[
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3], dtype=torch.int64),
            }
            for _ in range(num_samples)
        ],
    )


class _FakeData:
    def __init__(self, num_samples: int = 2) -> None:
        self.tensor = torch.zeros(num_samples, 3, 64, 64)
        self.sample_ids = [f"img_{i}.jpg" for i in range(num_samples)]


def test_assess_transparency_routes_detection_kind_to_explain_detection(
    tmp_path: Path,
) -> None:
    """ForwardOutput(detection) → ``_assess_transparency_detection`` →
    ``explain_detection`` is called once per configured explainer with the
    correct backend, inputs, and forward_output."""
    config = AppConfig(experiment_name="routing-test")
    set_output_root(config, tmp_path)
    config.transparency = {
        "ig_det": TransparencyConfig(
            _target_="CaptumExplainer",
            algorithm="IntegratedGradients",
            call={"target": 0},
            raitap={"detection": {"score_threshold": 0.5, "max_boxes": 5}},
            visualisers=[{"_target_": "DetectionImageVisualiser"}],
        )
    }

    model = _FakeModel()
    data = _FakeData(num_samples=2)
    forward = _make_detection_forward_output(num_samples=2)

    captured: dict[str, Any] = {}

    def fake_explain_detection(**kwargs: Any) -> Iterator[ExplanationResult]:
        captured.update(kwargs)
        # Yield one fake result per sample so we can assert downstream flatten.
        from raitap.transparency.results import ExplanationResult as _Result

        for _ in range(forward.batch_size):
            r = _Result.__new__(_Result)
            r.attributions = torch.zeros(1, 3, 64, 64)
            r.inputs = torch.zeros(1, 3, 64, 64)
            r.visualisers = []
            yield r

    class _FakeExplainer:
        algorithm = "IntegratedGradients"

        def check_backend_compat(self, backend: object) -> None:
            return None

    with (
        patch(
            "raitap.pipeline.phases.assess_transparency.Explanation",
            side_effect=AssertionError("classification path should not run"),
        ),
        patch(
            "raitap.transparency.factory.create_explainer",
            return_value=(_FakeExplainer(), "raitap.transparency.CaptumExplainer"),
        ),
        patch("raitap.transparency.factory.create_visualisers", return_value=[]),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_payload_compat",
            return_value=None,
        ),
        patch(
            "raitap.transparency.factory.check_explainer_visualiser_semantic_compat",
            return_value=None,
        ),
        patch(
            "raitap.pipeline.phases.explain_detection.explain_detection",
            side_effect=fake_explain_detection,
        ),
    ):
        explanations, _visualisations = assess_transparency(
            config,
            model,  # type: ignore[arg-type]
            data,  # type: ignore[arg-type]
            forward,
            input_metadata=None,
            resolved_preprocessing=None,
        )

    # The routing reached explain_detection with the right kwargs:
    assert captured["backend"] is model.backend
    assert captured["forward_output"] is forward
    assert captured["explainer_name"] == "ig_det"
    assert captured["explainer_target"] == "raitap.transparency.CaptumExplainer"
    assert "raitap_kwargs" in captured
    assert "call_kwargs" in captured
    # And the fake results were flattened into the explanations list:
    assert len(explanations) == 2


def test_assess_transparency_detection_skips_when_no_explainers(tmp_path: Path) -> None:
    config = AppConfig(experiment_name="empty")
    set_output_root(config, tmp_path)
    forward = _make_detection_forward_output(num_samples=1)

    explanations, visualisations = assess_transparency(
        config,
        _FakeModel(),  # type: ignore[arg-type]
        _FakeData(num_samples=1),  # type: ignore[arg-type]
        forward,
        input_metadata=None,
        resolved_preprocessing=None,
    )
    assert explanations == []
    assert visualisations == []
