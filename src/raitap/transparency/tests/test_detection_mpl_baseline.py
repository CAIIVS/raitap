"""MPL regression for DetectionImageVisualiser (house renderer, synthetic input).

Self-contained: no shap/captum, no model — builds the figure directly from a
deterministic synthetic attribution so the detection layout + sign-aware house
rendering is pixel-regressed. Library-specific (shap/captum) detection baselines
are deferred to a CI follow-up (need optional extras on the Linux image).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import torch

from raitap.transparency.contracts import DetectionBox, VisualisationContext
from raitap.transparency.visualisers.detection_image_visualiser import DetectionImageVisualiser

if TYPE_CHECKING:
    from matplotlib.figure import Figure

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="pytest-mpl baselines are generated on ubuntu-24.04; pixel diffs are not portable",
)


def _deterministic_detection_figure() -> Figure:
    # deterministic synthetic data — no RNG, fully reproducible
    grid = np.linspace(-1.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    attr = torch.from_numpy(np.stack([grid, -grid, grid * 0.5]))  # (3,16,16) signed
    img = torch.from_numpy(
        np.clip(np.stack([grid, grid, grid]) * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
    )  # (3,16,16) in [0,1]
    ctx = VisualisationContext(
        algorithm="HouseDemo",
        sample_names=None,
        show_sample_names=False,
        detection_box=DetectionBox(0, 0, (2.0, 2.0, 12.0, 12.0), 0.87, 1, "cat"),
        source_library=None,  # -> RaitapHouseRenderer (no shap/captum dependency)
        method_families=frozenset(),
    )
    return DetectionImageVisualiser().visualise(attr, inputs=img, context=ctx)


@pytest.mark.e2e
@pytest.mark.mpl
@pytest.mark.visual
@pytest.mark.mpl_image_compare(
    baseline_dir="mpl_baseline",
    filename="detection_house_heat_map.png",
    remove_text=True,
    savefig_kwargs={"dpi": 150},
    tolerance=2,
)
def test_detection_house_visual_regression() -> Figure:
    return _deterministic_detection_figure()
