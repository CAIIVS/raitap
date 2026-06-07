"""SHAP modern __call__ path end-to-end (#267)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    InputKind,
    InputSpec,
    TensorLayout,
)
from raitap.transparency.explainers.shap_explainer import ShapExplainer

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _predict_model() -> Callable[[torch.Tensor], torch.Tensor]:
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 5)).eval()

    def f(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return net(x.float())

    return f


@pytest.mark.usefixtures("needs_shap")
def test_partition_modern_tabular_returns_input_shaped() -> None:
    explainer = ShapExplainer(algorithm="PartitionExplainer")
    inputs = torch.randn(2, 8)
    spec = InputSpec(
        kind=InputKind.TABULAR,
        shape=(2, 8),
        layout=TensorLayout.BATCH_FEATURE,
        feature_names=None,
        metadata=None,
    )
    attrs = explainer.compute_attributions(
        _predict_model(),
        inputs,
        background_data=torch.randn(20, 8),
        target=0,
        input_spec=spec,
    )
    assert attrs.shape == (2, 8)
    assert attrs.dtype == torch.float32


def _image_predict_model() -> Callable[[torch.Tensor], torch.Tensor]:
    net = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 5),
    ).eval()

    def f(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return net(x.float())

    return f


@pytest.mark.usefixtures("needs_shap")
def test_partition_modern_image_returns_nchw() -> None:
    # Exercises the per-modality Image masker (opencv inpaint) + NCHW<->NHWC round-trip.
    pytest.importorskip("cv2")  # opencv backs the image masker
    explainer = ShapExplainer(algorithm="PartitionExplainer")
    inputs = torch.rand(1, 3, 8, 8)
    spec = InputSpec(
        kind=InputKind.IMAGE,
        shape=(1, 3, 8, 8),
        layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
        feature_names=None,
        metadata=None,
    )
    attrs = explainer.compute_attributions(
        _image_predict_model(),
        inputs,
        background_data=inputs,
        target=0,
        input_spec=spec,
        max_evals=100,
    )
    assert attrs.shape == (1, 3, 8, 8)  # NCHW, matches input
    assert attrs.dtype == torch.float32


@pytest.mark.usefixtures("needs_shap")
def test_partition_modern_explain_persists_input_features(tmp_path: Path) -> None:
    # Full explain() chokepoint: input_spec threads through, result is input-shaped.
    explainer = ShapExplainer(algorithm="PartitionExplainer")
    inputs = torch.randn(2, 8)
    result = explainer.explain(
        _predict_model(),
        inputs,
        run_dir=str(tmp_path),
        raitap_kwargs={
            "input_metadata": InputSpec(
                kind=InputKind.TABULAR,
                shape=(2, 8),
                layout=TensorLayout.BATCH_FEATURE,
                feature_names=None,
                metadata=None,
            )
        },
        background_data=torch.randn(20, 8),
        target=0,
    )
    assert result.semantics.output_space.space is ExplanationOutputSpace.INPUT_FEATURES
    assert result.attributions.shape == (2, 8)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["semantics"]["output_space"]["space"] == "input_features"
