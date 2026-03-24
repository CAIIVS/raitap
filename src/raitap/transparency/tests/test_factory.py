from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from omegaconf import OmegaConf

from raitap.transparency.factory import Explanation, create_visualisers
from raitap.transparency.methods_registry import VisualiserIncompatibilityError
from raitap.transparency.results import ExplanationResult

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig


def _make_config(tmp_path: Path, transparency_config: Any) -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            experiment_name="test",
            fallback_output_dir=str(tmp_path),
            transparency={"test_explainer": transparency_config},
        ),
    )


def test_explanation_returns_explanation_result(
    simple_cnn: torch.nn.Module, sample_images: torch.Tensor, tmp_path: Path
) -> None:
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.CaptumExplainer",
                "algorithm": "Saliency",
                "visualisers": [{"_target_": "raitap.transparency.CaptumImageVisualiser"}],
            }
        ),
    )

    model = SimpleNamespace(network=simple_cnn)
    explanation = Explanation(config, "test_explainer", model, sample_images, target=0)  # type: ignore[arg-type]

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency" / "test_explainer"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()

    visualisations = [
        explanation.visualise(visualiser)
        for visualiser in create_visualisers(config.transparency["test_explainer"])
    ]
    assert len(visualisations) == 1
    assert (explanation.run_dir / "CaptumImageVisualiser.png").exists()


def test_explanation_validates_visualisers_before_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DummyExplainer:
        algorithm = "KernelExplainer"

        def __init__(self) -> None:
            self.explain_called = False

        def explain(self, *args: Any, **kwargs: Any) -> None:
            self.explain_called = True
            raise AssertionError("explain() should not be called for incompatible visualisers")

    dummy_explainer = DummyExplainer()
    config = _make_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.transparency.ShapExplainer",
                "algorithm": "KernelExplainer",
                "visualisers": [{"_target_": "raitap.transparency.ShapImageVisualiser"}],
            }
        ),
    )

    monkeypatch.setattr(
        "raitap.transparency.factory.create_explainer",
        lambda _config: (dummy_explainer, "raitap.transparency.ShapExplainer"),
    )

    with pytest.raises(VisualiserIncompatibilityError):
        model = SimpleNamespace(network=torch.nn.Identity())
        Explanation(
            config,
            "test_explainer",
            model=model,  # type: ignore[arg-type]
            inputs=torch.zeros(1, 3, 8, 8),
        )

    assert dummy_explainer.explain_called is False
