from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from omegaconf import OmegaConf

from raitap.configs.schema import AppConfig
from raitap.transparency.factory import Explanation, create_visualisers
from raitap.transparency.methods_registry import VisualiserIncompatibilityError
from raitap.transparency.results import ExplanationResult


def _make_config(tmp_path, transparency_config) -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            experiment_name="test",
            fallback_output_dir=str(tmp_path),
            transparency=transparency_config,
        ),
    )


def test_explanation_returns_explanation_result(needs_captum, simple_cnn, sample_images, tmp_path):
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

    explanation = Explanation(config, simple_cnn, sample_images, target=0)

    assert isinstance(explanation, ExplanationResult)
    assert explanation.attributions.shape == sample_images.shape
    assert explanation.run_dir == tmp_path / "transparency"
    assert (explanation.run_dir / "attributions.pt").exists()
    assert (explanation.run_dir / "metadata.json").exists()

    visualisations = [
        explanation.visualise(visualiser) for visualiser in create_visualisers(config)
    ]
    assert len(visualisations) == 1
    assert (explanation.run_dir / "CaptumImageVisualiser.png").exists()


def test_explanation_validates_visualisers_before_compute(monkeypatch, tmp_path):
    class DummyExplainer:
        algorithm = "KernelExplainer"

        def __init__(self) -> None:
            self.explain_called = False

        def explain(self, *args, **kwargs):
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
        Explanation(
            config,
            model=torch.nn.Identity(),
            inputs=torch.zeros(1, 3, 8, 8),
        )

    assert dummy_explainer.explain_called is False
