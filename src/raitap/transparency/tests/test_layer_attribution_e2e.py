"""End-to-end test for captum Layer* attribution (#267).

The ``LayerActivationVisualiser`` render logic is covered by
``visualisers/tests/test_layer_activation_visualiser.py``. A pixel-regression
visual baseline is deferred to the ``regen-baselines.yml`` workflow (baselines
are only valid when generated on the canonical ubuntu image; see #194).
"""

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
from raitap.transparency.explainers.captum_explainer import CaptumExplainer

if TYPE_CHECKING:
    from pathlib import Path


def _model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 5),
    ).eval()


@pytest.mark.usefixtures("needs_captum")
def test_layer_conductance_persists_layer_activation(tmp_path: Path) -> None:
    model = _model()
    explainer = CaptumExplainer(algorithm="LayerConductance", layer_path="0")  # the Conv2d
    inputs = torch.randn(2, 3, 8, 8)
    result = explainer.explain(
        model,
        inputs,
        run_dir=str(tmp_path),
        raitap_kwargs={
            "input_metadata": InputSpec(
                kind=InputKind.IMAGE,
                shape=(2, 3, 8, 8),
                layout=TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH,
                feature_names=None,
                metadata=None,
            )
        },
        target=0,
    )
    assert result.semantics.output_space.space is ExplanationOutputSpace.LAYER_ACTIVATION
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["semantics"]["output_space"]["space"] == "layer_activation"
