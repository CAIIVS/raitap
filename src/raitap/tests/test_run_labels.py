from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from raitap.metrics import resolve_metric_targets
from raitap.run import pipeline as run_pipeline

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.data import Data
    from raitap.models import Model


class _BackendStub:
    def __init__(self, output: torch.Tensor) -> None:
        self._output = output

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        del inputs
        return self._output


def _minimal_run_config() -> SimpleNamespace:
    return SimpleNamespace(
        transparency={"default": {}},
        metrics=SimpleNamespace(_target_="ClassificationMetrics", num_classes=None),
    )


def test_resolve_metric_targets_uses_labels_when_available() -> None:
    predictions = torch.tensor([0, 1, 2])
    labels = torch.tensor([2, 1, 0])

    targets = resolve_metric_targets(predictions, labels)

    assert torch.equal(targets, labels)


def test_resolve_metric_targets_warns_and_falls_back_without_labels() -> None:
    predictions = torch.tensor([0, 1, 2])

    with pytest.warns(UserWarning, match="No ground-truth labels"):
        targets = resolve_metric_targets(predictions, None)

    assert torch.equal(targets, predictions)


def test_resolve_metric_targets_warns_and_falls_back_on_length_mismatch() -> None:
    predictions = torch.tensor([0, 1, 2])
    labels = torch.tensor([1, 0])

    with pytest.warns(UserWarning, match="do not match prediction count"):
        targets = resolve_metric_targets(predictions, labels)

    assert torch.equal(targets, predictions)


def test_run_without_tracking_passes_ground_truth_labels_to_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _minimal_run_config()
    logits = torch.tensor([[0.2, 0.8], [0.9, 0.1]], dtype=torch.float32)
    model = SimpleNamespace(backend=_BackendStub(logits))
    data = SimpleNamespace(
        tensor=torch.zeros((2, 3, 4, 4), dtype=torch.float32),
        labels=torch.tensor([1, 0], dtype=torch.long),
    )
    captured: dict[str, torch.Tensor] = {}

    class DummyExplanation:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def visualise(self) -> list[object]:
            return []

    def fake_metrics(
        _config: object,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> object:
        captured["predictions"] = predictions
        captured["targets"] = targets
        return SimpleNamespace()

    monkeypatch.setattr(run_pipeline, "metrics_run_enabled", lambda _cfg: True)
    monkeypatch.setattr(run_pipeline, "Metrics", fake_metrics)
    monkeypatch.setattr(run_pipeline, "Explanation", DummyExplanation)

    outputs = run_pipeline._run_without_tracking(
        cast("AppConfig", cast("object", config)),
        cast("Model", cast("object", model)),
        cast("Data", cast("object", data)),
    )

    assert outputs.metrics is not None
    assert torch.equal(captured["predictions"], torch.tensor([1, 0]))
    assert torch.equal(captured["targets"], data.labels)
