from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import torch

from raitap.transparency.results import ExplanationResult, _serialisable, resolve_default_run_dir

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def _make_explanation(run_dir: Path) -> ExplanationResult:
    return ExplanationResult(
        attributions=torch.randn(1, 3, 4, 4),
        inputs=torch.randn(1, 3, 4, 4),
        run_dir=run_dir,
        experiment_name="test-exp",
        explainer_target="raitap.transparency.CaptumExplainer",
        algorithm="Saliency",
        explainer_name="exp",
    )


def test_resolve_default_run_dir_uses_hydra_runtime(monkeypatch: MonkeyPatch) -> None:
    class _HydraRuntime:
        output_dir = "hydra-output"

    class _HydraConfig:
        runtime = _HydraRuntime()

    monkeypatch.setattr("raitap.transparency.results.HydraConfig.get", lambda: _HydraConfig())
    assert resolve_default_run_dir() == Path("hydra-output") / "transparency"


def test_resolve_default_run_dir_falls_back_to_cwd(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "raitap.transparency.results.HydraConfig.get",
        lambda: (_ for _ in ()).throw(ValueError("not initialised")),
    )
    assert resolve_default_run_dir() == Path.cwd() / "transparency"


def test_serialisable_handles_runtimeerror_from_item() -> None:
    tensor = torch.randn(2, 2)
    result = _serialisable(tensor)
    assert isinstance(result, str)
    assert "tensor" in result


def test_serialisable_converts_nested_set_and_dict() -> None:
    value = {"tags": {"a", "b"}, "nested": {"k": 1}}
    result = _serialisable(value)
    assert sorted(result["tags"]) == ["a", "b"]
    assert result["nested"]["k"] == 1


def test_explanation_log_no_visualisers_logs_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp"
    explanation = _make_explanation(run_dir)
    explanation.write_artifacts()
    tracker = MagicMock()

    explanation.log(tracker, artifact_path="transparency", use_subdirectory=True)

    tracker.log_artifacts.assert_called_once_with(
        run_dir,
        target_subdirectory="transparency/exp",
    )


def test_explanation_log_with_visualisers_stages_explanation_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp"
    explanation = _make_explanation(run_dir)
    explanation.write_artifacts()
    explanation.visualiser_targets = ["raitap.transparency.CaptumImageVisualiser"]
    explanation._metadata = MagicMock(return_value={"visualisers": []})  # type: ignore[method-assign]
    tracker = MagicMock()

    explanation.log(tracker, artifact_path="transparency", use_subdirectory=False)

    explanation._metadata.assert_called_once_with(visualiser_targets=[])  # type: ignore[attr-defined]
    tracker.log_artifacts.assert_called_once()
