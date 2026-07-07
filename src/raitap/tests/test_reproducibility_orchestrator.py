"""Orchestrator fan-out for the reproducibility caveat (issue #251).

The output-dir note and the CLI warning must fire whenever a stochastic method
ran, independent of reporting. These tests patch the heavy model/data
construction and drive ``_run_pipeline`` with a fake stochastic run.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from raitap.pipeline import orchestrator
from raitap.pipeline.outputs import ForwardOutput, RunOutputs
from raitap.reproducibility import REPRODUCIBILITY_FILENAME
from raitap.testing import make_app_config
from raitap.types import TaskKind

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


def _fo() -> ForwardOutput:
    return ForwardOutput(
        task_kind=TaskKind.classification,
        batch_size=1,
        payload=torch.tensor([[0.1, 0.9]]),
    )


def _outputs(*, stochastic: bool) -> RunOutputs:
    explanation = SimpleNamespace(
        name="shap_grad",
        algorithm="GradientExplainer",
        semantics=SimpleNamespace(seeding="global_rng" if stochastic else "deterministic"),
    )
    transparency_phase = SimpleNamespace(explanations=[explanation], report_order=10)
    return RunOutputs(
        forward_output=_fo(),
        phase_results={"transparency": transparency_phase},  # type: ignore[dict-item]
    )


def _patch_pipeline(monkeypatch: MonkeyPatch, run_dir: Path, outputs: RunOutputs) -> None:
    fake_model = SimpleNamespace(backend=SimpleNamespace(task_kind=TaskKind.classification))
    monkeypatch.setattr(orchestrator, "resolve_preprocessing", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "Model", lambda *a, **k: fake_model)
    monkeypatch.setattr(orchestrator, "Data", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(orchestrator, "_validate_report_sample_selection", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "run_phases", lambda *a, **k: outputs)
    monkeypatch.setattr(orchestrator, "reporting_enabled", lambda config: False)
    monkeypatch.setattr(orchestrator, "resolve_run_dir", lambda config: run_dir)


def test_stochastic_run_writes_md_and_warns_with_reporting_off(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config = make_app_config(
        experiment_name="repro",
        model={"source": "resnet50"},
        seed=None,
    )
    _patch_pipeline(monkeypatch, tmp_path, _outputs(stochastic=True))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        orchestrator._run_pipeline(config, verbose=False)  # type: ignore[arg-type]

    assert (tmp_path / REPRODUCIBILITY_FILENAME).exists()
    assert any("not bit-reproducible" in str(w.message) for w in caught)


def test_deterministic_run_emits_nothing(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    config = make_app_config(
        experiment_name="repro",
        model={"source": "resnet50"},
        seed=None,
    )
    _patch_pipeline(monkeypatch, tmp_path, _outputs(stochastic=False))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        orchestrator._run_pipeline(config, verbose=False)  # type: ignore[arg-type]

    assert not (tmp_path / REPRODUCIBILITY_FILENAME).exists()
    assert not any("not bit-reproducible" in str(w.message) for w in caught)


def test_seed_config_pins_global_seed(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(orchestrator, "pin_global_seed", lambda s: calls.append(s))
    config = make_app_config(
        experiment_name="repro",
        model={"source": "resnet50"},
        seed=99,
    )
    _patch_pipeline(monkeypatch, tmp_path, _outputs(stochastic=False))

    orchestrator._run_pipeline(config, verbose=False)  # type: ignore[arg-type]

    assert calls == [99]
