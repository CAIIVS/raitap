"""SHAP modern __call__ path end-to-end (#267)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

from raitap.transparency.contracts import InputKind, InputSpec, TensorLayout
from raitap.transparency.explainers.shap_explainer import ShapExplainer

if TYPE_CHECKING:
    from collections.abc import Callable


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
