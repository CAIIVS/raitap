from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

import raitap.pipeline.orchestrator as run_pipeline

# Bind the phase submodules as objects so ``monkeypatch.setattr`` patches the module
# directly — the lazy adapter ``__getattr__`` on ``raitap.metrics`` / ``raitap.transparency``
# makes dotted-string monkeypatch targets unreliable once another test unbinds them.
from raitap.configs.schema import MulticlassClassificationMetricsConfig
from raitap.metrics import phase as _metrics_phase
from raitap.metrics import resolve_metric_targets
from raitap.models.base_backend import ModelBackend
from raitap.testing import make_app_config
from raitap.transparency import phase as _transparency_phase
from raitap.types import Capability, TaskKind

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig

    from raitap.configs.schema import AppConfig
    from raitap.data import Data
    from raitap.models import Model
    from raitap.transparency.contracts import ExplainerAdapter


class _BackendStub(ModelBackend):
    def __init__(self, output: torch.Tensor) -> None:
        self._output = output

    @property
    def hardware_label(self) -> str:
        return "CPU"

    @property
    def task_kind(self) -> TaskKind:
        return TaskKind.classification

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def __call__(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del inputs, kwargs
        return self._output

    def autograd_module(self) -> torch.nn.Module:
        return torch.nn.Identity()


def _minimal_run_config() -> DictConfig:
    return make_app_config(
        transparency={"default": {}},
        metrics=MulticlassClassificationMetricsConfig(num_classes=3),
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


def test_run_phases_passes_ground_truth_labels_to_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _minimal_run_config()
    logits = torch.tensor([[0.2, 0.8], [0.9, 0.1]], dtype=torch.float32)
    model = SimpleNamespace(backend=_BackendStub(logits))
    data = SimpleNamespace(
        tensor=torch.zeros((2, 3, 4, 4), dtype=torch.float32),
        sample_ids=None,
        labels=torch.tensor([1, 0], dtype=torch.long),
    )
    captured: dict[str, torch.Tensor] = {}

    class _DummyResult:
        def _visualise(self) -> list[object]:
            return []

    class _DummyExplainer:
        def explain(self, *_args: object, **_kwargs: object) -> _DummyResult:
            return _DummyResult()

        def required_capabilities(self) -> frozenset[Capability]:
            return frozenset({Capability.AUTOGRAD})

    def _fake_prepare(
        cfg: object,
        name: str,
        _model: object,
        **_kwargs: object,
    ) -> _transparency_phase.PreparedExplainer:
        return _transparency_phase.PreparedExplainer(
            name=name,
            explainer=cast("ExplainerAdapter", _DummyExplainer()),
            explainer_target="raitap.transparency.Fake",
            visualisers=[],
            merged_kwargs={},
            raitap_kwargs={},
            call_provenance={},
            base_run_dir=cast("Path", None),
            backend=model.backend,
            experiment_name="test",
            explainer_config=cfg.transparency[name],  # type: ignore[attr-defined]
            class_names=None,
        )

    def fake_metrics(
        _config: object,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> object:
        captured["predictions"] = predictions
        captured["targets"] = targets
        return SimpleNamespace()

    monkeypatch.setattr(_metrics_phase, "metrics_run_enabled", lambda _cfg: True)
    monkeypatch.setattr(_metrics_phase, "Metrics", fake_metrics)
    monkeypatch.setattr(_transparency_phase, "prepare_explainer", _fake_prepare)

    outputs = run_pipeline.run_phases(
        cast("AppConfig", config),
        cast("Model", cast("object", model)),
        cast("Data", cast("object", data)),
    )

    assert "metrics" in outputs
    assert torch.equal(captured["predictions"], torch.tensor([1, 0]))
    assert torch.equal(captured["targets"], data.labels)
