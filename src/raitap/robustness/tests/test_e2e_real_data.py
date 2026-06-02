"""End-to-end smoke tests that exercise code paths a real Data loader hits.

The other unit tests in this package use small contiguous tensors built in the
test, paired with transparency, and skip the reporting layer. That misses
three regressions reviewer @pizzajojo found in PR #119:

* RAITAP's image loader produces non-contiguous NCHW tensors via HWC->CHW;
  torchattacks' ``PGDL2`` calls ``.view(...)`` and crashes.
* The image-pair visualiser's perturbation column is RGB-shaped, so ``imshow``
  ignored the diverging cmap and clipped tiny signed deltas to ``[0, 1]``.
* ``pipeline._run_without_tracking`` raised when transparency was empty even
  if robustness was configured, blocking robustness-only runs.

Each test guards one regression.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from raitap.models.backend import ModelBackend
from raitap.pipeline.orchestrator import run_without_tracking as _run_without_tracking
from raitap.robustness import RobustnessAssessment, RobustnessResult
from raitap.testing import make_pixel_linear_classifier

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig
    from raitap.models import Model

torchattacks = pytest.importorskip("torchattacks")


class _BackendStub(ModelBackend):
    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model
        self.supports_torch_autograd = True

    @property
    def hardware_label(self) -> str:
        return "stub"

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._model(inputs)

    def as_model_for_explanation(self) -> torch.nn.Module:
        return self._model


def _hwc_to_nchw_non_contiguous(batch: int = 2, hw: int = 8) -> torch.Tensor:
    """Build an NCHW tensor via the same HWC->CHW transpose the loader uses.

    Returns a strided (non-contiguous) view to mirror what
    ``raitap.data`` produces from a real image batch.
    """
    hwc = torch.rand(batch, hw, hw, 3)
    nchw = hwc.permute(0, 3, 1, 2)  # contiguous=False
    assert not nchw.is_contiguous()
    return nchw


def _make_robustness_config(tmp_path: Path, assessor_cfg: Any) -> AppConfig:
    return cast(
        "AppConfig",
        SimpleNamespace(
            experiment_name="test",
            _output_root=str(tmp_path),
            transparency={},
            robustness={"pgd": assessor_cfg},
        ),
    )


def test_torchattacks_pgdl2_handles_non_contiguous_input(tmp_path: Path) -> None:
    """Regression: PGDL2 used to crash on RAITAP loader's HWC->CHW tensors."""
    inputs = _hwc_to_nchw_non_contiguous()
    targets = torch.tensor([0, 1])
    model = make_pixel_linear_classifier(hw=8)

    config = _make_robustness_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.robustness.TorchattacksAssessor",
                "algorithm": "PGDL2",
                "constructor": {"eps": 0.1, "alpha": 0.02, "steps": 3},
                "visualisers": [],
            }
        ),
    )
    raitap_model = cast("Model", SimpleNamespace(backend=_BackendStub(model)))

    result = RobustnessAssessment(  # type: ignore[arg-type]
        config,
        "pgd",
        raitap_model,
        inputs,
        targets,
    )

    assert isinstance(result, RobustnessResult)
    assert result.perturbed_inputs is not None
    assert result.perturbed_inputs.shape == inputs.shape


def test_image_pair_visualiser_diff_uses_diverging_cmap(tmp_path: Path) -> None:
    """Regression: diff axis used to receive RGB and silently dropped cmap/vmin/vmax."""
    inputs = _hwc_to_nchw_non_contiguous(batch=2, hw=8)
    targets = torch.tensor([0, 1])
    model = make_pixel_linear_classifier(hw=8)

    config = _make_robustness_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.robustness.TorchattacksAssessor",
                "algorithm": "FGSM",
                "constructor": {"eps": 0.05},
                "visualisers": [
                    {"_target_": "raitap.robustness.ImagePairVisualiser"},
                ],
            }
        ),
    )
    raitap_model = cast("Model", SimpleNamespace(backend=_BackendStub(model)))

    result = RobustnessAssessment(  # type: ignore[arg-type]
        config,
        "pgd",
        raitap_model,
        inputs,
        targets,
    )
    visualisations = result.visualise()
    assert visualisations, "expected at least one rendered visualisation"
    figure = visualisations[0].figure
    try:
        # Layout is N rows x 3 columns; the third column is the diff axis.
        diff_axis = figure.axes[2]
        assert diff_axis.images, "diff axis missing imshow output"
        diff_image = diff_axis.images[0]
        # Diverging cmap honoured (i.e. the array reached imshow as 2D scalars).
        assert diff_image.get_cmap().name == "RdBu_r"
        rendered = diff_image.get_array()
        assert rendered is not None
        assert rendered.ndim == 2, (
            "perturbation panel must be a 2D scalar map; "
            "RGB arrays defeat the diverging cmap and clip signed deltas to [0,1]."
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_assess_rejects_empty_batch(tmp_path: Path) -> None:
    """Empty batches must error loudly rather than crash deep in matplotlib / metrics."""
    inputs = torch.empty(0, 3, 8, 8)
    targets = torch.empty(0, dtype=torch.long)
    model = make_pixel_linear_classifier(hw=8)

    config = _make_robustness_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.robustness.TorchattacksAssessor",
                "algorithm": "FGSM",
                "constructor": {"eps": 0.05},
                "visualisers": [],
            }
        ),
    )
    raitap_model = cast("Model", SimpleNamespace(backend=_BackendStub(model)))

    with pytest.raises(ValueError, match="empty batch"):
        RobustnessAssessment(  # type: ignore[arg-type]
            config,
            "pgd",
            raitap_model,
            inputs,
            targets,
        )


def test_pipeline_allows_robustness_only_runs(tmp_path: Path) -> None:
    """Regression: pipeline used to require at least one explainer."""
    inputs = _hwc_to_nchw_non_contiguous(batch=2, hw=8)
    targets = torch.tensor([0, 1])
    model = make_pixel_linear_classifier(hw=8)

    config = _make_robustness_config(
        tmp_path,
        OmegaConf.create(
            {
                "_target_": "raitap.robustness.TorchattacksAssessor",
                "algorithm": "FGSM",
                "constructor": {"eps": 0.05},
                "visualisers": [],
            }
        ),
    )
    # Add the bits ``_run_without_tracking`` reads.
    config.metrics = SimpleNamespace(_target_=None, num_classes=3)  # type: ignore[attr-defined]
    config.reporting = None  # type: ignore[attr-defined]

    raitap_model = cast("Model", SimpleNamespace(backend=_BackendStub(model)))
    data_stub = SimpleNamespace(
        tensor=inputs,
        labels=targets,
        sample_ids=["a", "b"],
        input_metadata=None,
    )
    data_stub.log = MagicMock()  # type: ignore[attr-defined]

    outputs = _run_without_tracking(config, raitap_model, data_stub)  # type: ignore[arg-type]

    assert "transparency" not in outputs
    assert len(outputs.robustness_results) == 1
    assert outputs.robustness_results[0].assessor_name == "pgd"
