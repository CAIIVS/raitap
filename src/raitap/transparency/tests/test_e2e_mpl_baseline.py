"""
Deterministic MPL regression test for transparency visual output.

This test intentionally returns a Matplotlib figure so `pytest-mpl` can compare
it against a committed PNG baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

from raitap.transparency.explainers import CaptumExplainer
from raitap.transparency.visualisers import CaptumImageVisualiser

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_BASELINE_DIR = Path(__file__).with_name("mpl_baseline")
_BASELINE_FILE = _BASELINE_DIR / "captum_ig_image_heat_map.png"
_MPL_TOLERANCE = 10 if sys.platform.startswith("win") else 2


def _require_parameter(parameter: torch.Tensor | None, *, name: str) -> torch.Tensor:
    if parameter is None:
        raise ValueError(f"{name} must be present for the deterministic MPL baseline model.")
    return parameter


def _build_deterministic_cnn() -> nn.Module:
    model = nn.Sequential(
        nn.Conv2d(3, 2, kernel_size=1, bias=True),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(2, 2, bias=True),
    )
    with torch.no_grad():
        conv = model[0]
        linear = model[4]
        assert isinstance(conv, nn.Conv2d)
        assert isinstance(linear, nn.Linear)

        conv.weight.copy_(
            torch.tensor(
                [
                    [[[0.70]], [[-0.25]], [[0.15]]],
                    [[[-0.35]], [[0.55]], [[0.40]]],
                ],
                dtype=torch.float32,
            )
        )
        conv_bias = _require_parameter(conv.bias, name="conv.bias")
        linear_bias = _require_parameter(linear.bias, name="linear.bias")
        conv_bias.copy_(torch.tensor([0.05, -0.10], dtype=torch.float32))
        linear.weight.copy_(torch.tensor([[0.60, -0.45], [-0.30, 0.80]], dtype=torch.float32))
        linear_bias.copy_(torch.tensor([0.10, -0.05], dtype=torch.float32))

    model.eval()
    return model


def _literal_image_batch() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [
                    [0.00, 0.10, 0.20, 0.30],
                    [0.05, 0.15, 0.25, 0.35],
                    [0.10, 0.20, 0.30, 0.40],
                    [0.15, 0.25, 0.35, 0.45],
                ],
                [
                    [0.45, 0.35, 0.25, 0.15],
                    [0.40, 0.30, 0.20, 0.10],
                    [0.35, 0.25, 0.15, 0.05],
                    [0.30, 0.20, 0.10, 0.00],
                ],
                [
                    [0.20, 0.30, 0.40, 0.50],
                    [0.25, 0.35, 0.45, 0.55],
                    [0.30, 0.40, 0.50, 0.60],
                    [0.35, 0.45, 0.55, 0.65],
                ],
            ]
        ],
        dtype=torch.float32,
    )


def _ensure_baseline_or_generation_mode(request: pytest.FixtureRequest) -> None:
    generate_path = getattr(request.config.option, "mpl_generate_path", None)
    if _BASELINE_FILE.exists() or generate_path:
        return

    pytest.fail(
        "The MPL baseline image is missing. Generate or provide the baseline "
        f"PNG at: {_BASELINE_FILE.as_posix()}\n\n"
        "Suggested command to regenerate candidate artifacts locally:\n"
        "uv run pytest src/raitap/transparency/tests/test_e2e_mpl_baseline.py "
        "-m e2e --mpl-generate-path=src/raitap/transparency/tests/mpl_baseline_candidate -v"
    )


@pytest.mark.e2e
@pytest.mark.mpl
@pytest.mark.usefixtures("needs_captum")
@pytest.mark.mpl_image_compare(
    baseline_dir="mpl_baseline",
    filename="captum_ig_image_heat_map.png",
    remove_text=True,
    savefig_kwargs={"dpi": 150},
    tolerance=_MPL_TOLERANCE,
)
def test_captum_integrated_gradients_image_heat_map_baseline(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Figure:
    _ensure_baseline_or_generation_mode(request)

    model = _build_deterministic_cnn()
    inputs = _literal_image_batch()
    explanation = CaptumExplainer("IntegratedGradients").explain(
        model,
        inputs,
        run_dir=tmp_path / "mpl_regression",
        target=1,
    )

    visualiser = CaptumImageVisualiser(
        method="heat_map",
        sign="all",
        show_colorbar=False,
        include_original_image=False,
    )
    figure = visualiser.visualise(explanation.attributions, inputs=inputs, max_samples=1)
    figure.set_size_inches(4.0, 4.0)
    figure.tight_layout()
    return figure
