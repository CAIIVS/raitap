"""Parametrised E2E coverage for every empirical robustness algorithm.

Each algorithm in :data:`MATRIX_CASES` runs once against a tiny shared
classifier on a tiny shared image batch — the test gates "does the assessor
wire up end-to-end and produce a perturbed-inputs tensor of the expected
shape" rather than attack effectiveness.

Marabou (formal verification) has its own dedicated tests
(``test_e2e_marabou_*.py``) — different contract, different fixtures.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from omegaconf import OmegaConf

from raitap.models.base_backend import ModelBackend
from raitap.robustness import RobustnessAssessment, RobustnessResult
from raitap.robustness.tests.e2e_assessor_matrix import MATRIX_CASES, AssessorMatrixCase
from raitap.testing import make_pixel_linear_classifier
from raitap.types import Capability

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig
    from raitap.models import Model


class _BackendStub(ModelBackend):
    """Minimal backend that wraps a torch nn.Module — no preprocessing, no
    Hydra config dance. Matches the shape used in ``test_e2e_real_data.py``."""

    provides = frozenset({Capability.AUTOGRAD})

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    @property
    def hardware_label(self) -> str:
        return "stub"

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs

    def __call__(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self._model(inputs)

    def autograd_module(self) -> torch.nn.Module:
        return self._model


@pytest.fixture
def tiny_inputs() -> torch.Tensor:
    # Deterministic seed → matrix cases reproducible across runs.
    gen = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, 8, 8, generator=gen)


@pytest.fixture
def tiny_targets() -> torch.Tensor:
    return torch.tensor([0, 1])


@pytest.fixture
def tiny_model() -> torch.nn.Module:
    return make_pixel_linear_classifier(num_classes=3)


def _make_robustness_config(tmp_path: Path, family: str, case: AssessorMatrixCase) -> AppConfig:
    target = (
        "raitap.robustness.TorchattacksAssessor"
        if family == "torchattacks"
        else "raitap.robustness.FoolboxAssessor"
    )
    return cast(
        "AppConfig",
        SimpleNamespace(
            experiment_name=f"test_matrix_{case.id}",
            _output_root=str(tmp_path),
            transparency={},
            robustness={
                "matrix": OmegaConf.create(
                    {
                        "_target_": target,
                        "algorithm": case.algorithm,
                        "constructor": dict(case.constructor_kwargs),
                        "call": dict(case.call_kwargs),
                        "visualisers": [],
                    }
                )
            },
        ),
    )


@pytest.mark.parametrize("case", MATRIX_CASES, ids=[c.id for c in MATRIX_CASES])
def test_assessor_matrix_happy_path(
    case: AssessorMatrixCase,
    tmp_path: Path,
    tiny_inputs: torch.Tensor,
    tiny_targets: torch.Tensor,
    tiny_model: torch.nn.Module,
) -> None:
    if importlib.util.find_spec(case.needs_extra) is None:
        pytest.skip(f"{case.needs_extra} extra not installed")

    config = _make_robustness_config(tmp_path, case.family, case)
    raitap_model = cast("Model", SimpleNamespace(backend=_BackendStub(tiny_model)))

    result = RobustnessAssessment(
        config,
        "matrix",
        raitap_model,
        tiny_inputs,
        tiny_targets,
    )

    assert isinstance(result, RobustnessResult), f"{case.id}: bad return type"
    assert result.perturbed_inputs is not None, f"{case.id}: no perturbed_inputs"
    assert result.perturbed_inputs.shape == tiny_inputs.shape, (
        f"{case.id}: perturbed shape {result.perturbed_inputs.shape} "
        f"!= inputs shape {tiny_inputs.shape}"
    )
