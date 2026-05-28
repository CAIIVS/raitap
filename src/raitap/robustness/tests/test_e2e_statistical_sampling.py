"""E2E: gaussian_noise on a tiny classifier through the RobustnessAssessment factory.

gaussian_noise is size-agnostic, so it works on small images (blur/weather
corruptions need larger inputs and are out of scope for this gate).
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from omegaconf import OmegaConf

from raitap.models.backend import ModelBackend
from raitap.robustness import RobustnessAssessment, RobustnessResult
from raitap.testing import make_pixel_linear_classifier

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig
    from raitap.models import Model


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


@pytest.mark.skipif(
    importlib.util.find_spec("imagecorruptions") is None,
    reason="imagecorruptions extra not installed",
)
def test_gaussian_noise_e2e(tmp_path: Path) -> None:
    gen = torch.Generator().manual_seed(0)
    inputs = torch.rand(6, 3, 32, 32, generator=gen)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])

    config = cast(
        "AppConfig",
        SimpleNamespace(
            experiment_name="e2e_sampling",
            _output_root=str(tmp_path),
            transparency={},
            robustness={
                "avg": OmegaConf.create(
                    {
                        "_target_": "raitap.robustness.ImageCorruptionsAssessor",
                        "algorithm": "gaussian_noise",
                        "constructor": {"severity": 3},
                        "call": {},
                        "raitap": {"ci_method": "wilson", "ci_level": 0.95},
                        "visualisers": [{"_target_": "CorruptionAccuracyVisualiser"}],
                    }
                )
            },
        ),
    )
    raitap_model = cast(
        "Model",
        SimpleNamespace(backend=_BackendStub(make_pixel_linear_classifier(num_classes=3, hw=32))),
    )

    result = RobustnessAssessment(config, "avg", raitap_model, inputs, targets)

    assert isinstance(result, RobustnessResult)
    assert result.metrics.n_samples == 6
    assert 0.0 <= result.metrics.corrupted_accuracy <= 1.0
    assert (
        result.metrics.accuracy_ci_low
        <= result.metrics.corrupted_accuracy
        <= result.metrics.accuracy_ci_high
    )
    assert (result.run_dir / "metadata.json").exists()
    assert (result.run_dir / "robustness_data.pt").exists()
