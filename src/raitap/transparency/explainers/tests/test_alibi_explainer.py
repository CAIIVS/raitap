from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from raitap.transparency.explainers.alibi_explainer import AlibiExplainer
from raitap.transparency.factory import Explanation

if TYPE_CHECKING:
    from pathlib import Path

    import torch


@pytest.mark.usefixtures("needs_alibi")
def test_alibi_kernel_shap_runs_with_torch_model(
    simple_mlp: torch.nn.Module,
    sample_tabular: torch.Tensor,
    tmp_path: Path,
) -> None:
    explainer = AlibiExplainer("KernelShap")
    inputs = sample_tabular[:4]
    result = explainer.explain(
        simple_mlp,
        inputs,
        run_dir=tmp_path / "transparency",
        nsamples=20,
    )
    assert result.attributions.shape == inputs.shape
    assert (tmp_path / "transparency" / "attributions.pt").exists()


@pytest.mark.usefixtures("needs_alibi")
def test_explanation_factory_alibi_emits_warning_once(
    simple_mlp: torch.nn.Module,
    sample_tabular: torch.Tensor,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from omegaconf import OmegaConf

    config = cast(
        "Any",
        SimpleNamespace(
            experiment_name="test",
            _output_root=str(tmp_path),
            transparency={
                "a1": OmegaConf.create(
                    {
                        "_target_": "raitap.transparency.AlibiExplainer",
                        "algorithm": "KernelShap",
                        "call": {"nsamples": 12},
                        "visualisers": [
                            {"_target_": "raitap.transparency.TabularBarChartVisualiser"}
                        ],
                    }
                ),
                "a2": OmegaConf.create(
                    {
                        "_target_": "raitap.transparency.AlibiExplainer",
                        "algorithm": "KernelShap",
                        "call": {"nsamples": 12},
                        "visualisers": [],
                    }
                ),
            },
        ),
    )
    model = SimpleNamespace(
        backend=SimpleNamespace(
            supports_torch_autograd=True,
            _prepare_kwargs=lambda k: k,
            as_model_for_explanation=lambda: simple_mlp,
        )
    )
    inputs = sample_tabular[:3]
    with caplog.at_level("WARNING"):
        Explanation(config, "a1", model, inputs)  # type: ignore[arg-type]
        Explanation(config, "a2", model, inputs)  # type: ignore[arg-type]

    alibi_warnings = [r for r in caplog.records if "Alibi Explain" in r.message]
    assert len(alibi_warnings) == 1
