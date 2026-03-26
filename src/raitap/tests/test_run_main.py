from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch

    from raitap.configs.schema import AppConfig

from raitap import run as run_module


class _FakeExplainerResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.log_calls: list[bool] = []
        self.visualise_calls = 0

    def visualise(self) -> list[_FakeVisualisationResult]:
        self.visualise_calls += 1
        return [_FakeVisualisationResult(self.name)]

    def log(self, tracker: object, use_subdirectory: bool = True) -> None:
        del tracker
        self.log_calls.append(use_subdirectory)


class _FakeVisualisationResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.log_calls: list[bool] = []

    def log(self, tracker: object, use_subdirectory: bool = True) -> None:
        del tracker
        self.log_calls.append(use_subdirectory)


def test_hydra_main_composes_default_config(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(config: object) -> None:
        captured["config"] = config

    monkeypatch.setattr(run_module, "run", fake_run)
    monkeypatch.setattr(run_module, "print_summary", lambda _cfg: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "raitap.run",
            f"hydra.run.dir={tmp_path / 'hydra-run'}",
            "hydra.output_subdir=null",
        ],
    )

    run_module.main()

    cfg = cast("AppConfig", captured["config"])
    assert cfg.model.source == "vit_b_32"
    assert cfg.metrics._target_ == "ClassificationMetrics"
    assert cfg.transparency


def test_run_without_tracking_returns_outputs(monkeypatch: MonkeyPatch) -> None:
    model = SimpleNamespace(network=torch.nn.Identity())
    data = SimpleNamespace(tensor=torch.randn(2, 3))
    fake_output = run_module.RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        predicted_classes=torch.tensor([0, 0]),
    )
    tracker_factory = MagicMock()

    monkeypatch.setattr(run_module, "Model", lambda _cfg: model)
    monkeypatch.setattr(run_module, "Data", lambda _cfg: data)
    monkeypatch.setattr(run_module, "_run_without_tracking", lambda _c, _m, _d: fake_output)
    monkeypatch.setattr(run_module.BaseTracker, "create_tracker", tracker_factory)

    config = SimpleNamespace(tracking=None)
    result = run_module.run(config)  # type: ignore[arg-type]

    assert result is fake_output
    tracker_factory.assert_not_called()


def test_run_with_tracking_logs_all_outputs(monkeypatch: MonkeyPatch) -> None:
    model = SimpleNamespace(network=torch.nn.Identity(), log=MagicMock())
    data = SimpleNamespace(tensor=torch.randn(2, 3), log=MagicMock())
    explanation = _FakeExplainerResult("exp1")
    visualisation = explanation.visualise()[0]
    metrics_eval = SimpleNamespace(log=MagicMock())
    fake_output = run_module.RunOutputs(
        explanations=[explanation],  # type: ignore[list-item]
        visualisations=[visualisation],  # type: ignore[list-item]
        metrics=metrics_eval,  # type: ignore[arg-type]
        predicted_classes=torch.tensor([0, 0]),
    )
    tracker = MagicMock()

    class _TrackerContext:
        def __enter__(self) -> object:
            return tracker

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type, exc, tb
            return False

    monkeypatch.setattr(run_module, "Model", lambda _cfg: model)
    monkeypatch.setattr(run_module, "Data", lambda _cfg: data)
    monkeypatch.setattr(run_module, "_run_without_tracking", lambda _c, _m, _d: fake_output)
    monkeypatch.setattr(run_module.BaseTracker, "create_tracker", lambda _cfg: _TrackerContext())

    config = SimpleNamespace(tracking=SimpleNamespace(_target_="MLFlowTracker", log_model=True))
    run_module.run(config)  # type: ignore[arg-type]

    tracker.log_config.assert_called_once()
    model.log.assert_called_once_with(tracker)
    data.log.assert_called_once_with(tracker)
    metrics_eval.log.assert_called_once_with(tracker)
    assert explanation.log_calls == [False]
    assert visualisation.log_calls == [False]


def test_run_with_tracking_skips_model_logging_when_disabled(monkeypatch: MonkeyPatch) -> None:
    model = SimpleNamespace(network=torch.nn.Identity(), log=MagicMock())
    data = SimpleNamespace(tensor=torch.randn(1, 3), log=MagicMock())
    explanation = _FakeExplainerResult("exp1")
    visualisation = explanation.visualise()[0]
    fake_output = run_module.RunOutputs(
        explanations=[explanation],  # type: ignore[list-item]
        visualisations=[visualisation],  # type: ignore[list-item]
        metrics=None,
        predicted_classes=torch.tensor([0]),
    )
    tracker = MagicMock()

    class _TrackerContext:
        def __enter__(self) -> object:
            return tracker

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type, exc, tb
            return False

    monkeypatch.setattr(run_module, "Model", lambda _cfg: model)
    monkeypatch.setattr(run_module, "Data", lambda _cfg: data)
    monkeypatch.setattr(run_module, "_run_without_tracking", lambda _c, _m, _d: fake_output)
    monkeypatch.setattr(run_module.BaseTracker, "create_tracker", lambda _cfg: _TrackerContext())

    config = SimpleNamespace(tracking=SimpleNamespace(_target_="MLFlowTracker", log_model=False))
    run_module.run(config)  # type: ignore[arg-type]

    model.log.assert_not_called()
    data.log.assert_called_once_with(tracker)


def test_run_with_multiple_explainers_uses_subdirs(monkeypatch: MonkeyPatch) -> None:
    model = SimpleNamespace(network=torch.nn.Identity(), log=MagicMock())
    data = SimpleNamespace(tensor=torch.randn(2, 3), log=MagicMock())
    exp1 = _FakeExplainerResult("exp1")
    vis1 = exp1.visualise()[0]
    exp2 = _FakeExplainerResult("exp2")
    vis2 = exp2.visualise()[0]
    fake_output = run_module.RunOutputs(
        explanations=[exp1, exp2],  # type: ignore[list-item]
        visualisations=[vis1, vis2],  # type: ignore[list-item]
        metrics=None,
        predicted_classes=torch.tensor([0, 0]),
    )
    tracker = MagicMock()

    class _TrackerContext:
        def __enter__(self) -> object:
            return tracker

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type, exc, tb
            return False

    monkeypatch.setattr(run_module, "Model", lambda _cfg: model)
    monkeypatch.setattr(run_module, "Data", lambda _cfg: data)
    monkeypatch.setattr(run_module, "_run_without_tracking", lambda _c, _m, _d: fake_output)
    monkeypatch.setattr(run_module.BaseTracker, "create_tracker", lambda _cfg: _TrackerContext())

    config = SimpleNamespace(tracking=SimpleNamespace(_target_="MLFlowTracker", log_model=False))
    run_module.run(config)  # type: ignore[arg-type]

    assert exp1.log_calls == [True]
    assert exp2.log_calls == [True]
    assert vis1.log_calls == [True]
    assert vis2.log_calls == [True]


def test_run_with_tracking_config_but_no_target_skips_tracking(monkeypatch: MonkeyPatch) -> None:
    model = SimpleNamespace(network=torch.nn.Identity())
    data = SimpleNamespace(tensor=torch.randn(2, 3))
    fake_output = run_module.RunOutputs(
        explanations=[],
        visualisations=[],
        metrics=None,
        predicted_classes=torch.tensor([0, 0]),
    )
    tracker_factory = MagicMock()

    monkeypatch.setattr(run_module, "Model", lambda _cfg: model)
    monkeypatch.setattr(run_module, "Data", lambda _cfg: data)
    monkeypatch.setattr(run_module, "_run_without_tracking", lambda _c, _m, _d: fake_output)
    monkeypatch.setattr(run_module.BaseTracker, "create_tracker", tracker_factory)

    # tracking config exists but _target_ is None or empty
    config = SimpleNamespace(tracking=SimpleNamespace(_target_=None))
    result = run_module.run(config)  # type: ignore[arg-type]

    assert result is fake_output
    tracker_factory.assert_not_called()


def test_run_without_tracking_raises_if_no_explainers(monkeypatch: MonkeyPatch) -> None:
    model = SimpleNamespace(network=torch.nn.Identity())
    data = SimpleNamespace(tensor=torch.randn(2, 3))
    config = SimpleNamespace(transparency={}, metrics=SimpleNamespace(num_classes=None))

    monkeypatch.setattr(run_module, "metrics_run_enabled", lambda _cfg: False)

    with pytest.raises(ValueError, match="No explainers configured"):
        run_module._run_without_tracking(config, model, data)  # type: ignore[arg-type]


def test_run_without_tracking_infers_num_classes_and_runs_metrics(monkeypatch: MonkeyPatch) -> None:
    class _Net(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            del x
            return torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])

    model = SimpleNamespace(network=_Net())
    data = SimpleNamespace(tensor=torch.randn(2, 4))
    explanation = _FakeExplainerResult("exp")
    metrics_calls: list[tuple[object, torch.Tensor, torch.Tensor]] = []

    def _fake_explanation(*_args: object, **_kwargs: object) -> _FakeExplainerResult:
        return explanation

    def _fake_metrics(cfg: object, preds: torch.Tensor, targs: torch.Tensor) -> object:
        metrics_calls.append((cfg, preds, targs))
        return SimpleNamespace()

    config = SimpleNamespace(
        transparency={"one": {}},
        metrics=SimpleNamespace(num_classes=None),
    )
    monkeypatch.setattr(run_module, "metrics_run_enabled", lambda _cfg: True)
    monkeypatch.setattr(run_module, "Explanation", _fake_explanation)
    monkeypatch.setattr(run_module, "Metrics", _fake_metrics)

    outputs = run_module._run_without_tracking(config, model, data)  # type: ignore[arg-type]

    assert config.metrics.num_classes == 3
    assert outputs.metrics is not None
    assert len(metrics_calls) == 1
    _, preds, targs = metrics_calls[0]
    assert torch.equal(preds, targs)
    assert explanation.visualise_calls == 1


def test_run_without_tracking_uses_provided_num_classes(monkeypatch: MonkeyPatch) -> None:
    class _Net(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            del x
            return torch.tensor([[0.1, 0.9, 0.0]])

    model = SimpleNamespace(network=_Net())
    data = SimpleNamespace(tensor=torch.randn(1, 4))
    explanation = _FakeExplainerResult("exp")

    def _fake_explanation(*_args: object, **_kwargs: object) -> _FakeExplainerResult:
        return explanation

    config = SimpleNamespace(
        transparency={"one": {}},
        metrics=SimpleNamespace(num_classes=10),
    )
    monkeypatch.setattr(run_module, "metrics_run_enabled", lambda _cfg: True)
    monkeypatch.setattr(run_module, "Explanation", _fake_explanation)
    monkeypatch.setattr(run_module, "Metrics", lambda _c, _p, _t: SimpleNamespace())

    run_module._run_without_tracking(config, model, data)  # type: ignore[arg-type]

    assert config.metrics.num_classes == 10
