from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from raitap.models.backend import ModelBackend
from raitap.robustness.assessors import TorchattacksAssessor
from raitap.robustness.contracts import MethodKind
from raitap.robustness.exceptions import MethodKindVisualiserIncompatibilityError
from raitap.robustness.factory import (
    RobustnessAssessment,
    _parse_assessor_config,
    _resolve_call_data_sources,
    check_assessor_visualiser_compat,
)
from raitap.robustness.results import ConfiguredRobustnessVisualiser, RobustnessResult
from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser

if TYPE_CHECKING:
    from pathlib import Path


class _BackendStub(ModelBackend):
    supports_torch_autograd = True

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

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


class _OnlyFormalVisualiser(BaseRobustnessVisualiser):
    supported_method_kinds = frozenset({MethodKind.FORMAL_VERIFICATION})

    def visualise(self, result, *, context, **kwargs) -> Any:  # noqa: ANN001
        raise NotImplementedError


def test_parse_validates_top_level_keys() -> None:
    with pytest.raises(ValueError, match="Unknown robustness assessor config keys"):
        _parse_assessor_config(
            {
                "_target_": "TorchattacksAssessor",
                "algorithm": "PGD",
                "wibble": True,  # unknown key
            }
        )


def test_parse_migrates_misplaced_raitap_keys() -> None:
    parsed = _parse_assessor_config(
        {
            "_target_": "TorchattacksAssessor",
            "algorithm": "PGD",
            "call": {"eps": 0.03, "batch_size": 8},
        }
    )
    # batch_size is RAITAP-owned; factory migrates it from `call` to `raitap`.
    assert "batch_size" not in parsed.call
    assert parsed.raitap["batch_size"] == 8


def test_resolve_call_data_sources_passes_through_non_source_dicts() -> None:
    out = _resolve_call_data_sources({"target_labels": [0, 1]})
    assert out == {"target_labels": [0, 1]}


def test_check_visualiser_compat_raises_on_method_kind_mismatch() -> None:
    assessor = TorchattacksAssessor(algorithm="PGD")  # EMPIRICAL_ATTACK
    visualiser = _OnlyFormalVisualiser()
    configured = [ConfiguredRobustnessVisualiser(visualiser=visualiser)]
    with pytest.raises(MethodKindVisualiserIncompatibilityError):
        check_assessor_visualiser_compat(
            assessor,
            "raitap.robustness.assessors.TorchattacksAssessor",
            configured,
        )


def test_robustness_uses_model_resolved_preprocessing_for_call_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Auxiliary robustness call-data should reuse the run-level preprocessing object."""

    class _ShapeModule(torch.nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return torch.zeros(3, 5, 5, dtype=image.dtype)

    class RecordingAssessor:
        method_kind = MethodKind.EMPIRICAL_ATTACK

        def __init__(self) -> None:
            self.last_assess_kwargs: dict[str, Any] = {}

        def check_backend_compat(self, backend: object) -> None:
            del backend

        def assess(self, *_args: Any, **kwargs: Any) -> RobustnessResult:
            self.last_assess_kwargs = dict(kwargs)
            return MagicMock(spec=RobustnessResult)

    from PIL import Image

    from raitap.data.preprocessing import ResolvedPreprocessing

    bg_dir = tmp_path / "bg"
    bg_dir.mkdir()
    for i in range(2):
        Image.fromarray(torch.zeros(8, 8, 3, dtype=torch.uint8).numpy(), "RGB").save(
            bg_dir / f"bg{i}.png"
        )

    assessor = RecordingAssessor()
    config = SimpleNamespace(
        experiment_name="test",
        _output_root=str(tmp_path),
        model=SimpleNamespace(source="resnet50"),
        data=SimpleNamespace(preprocessing="model-bundled"),
        robustness={
            "pgd": OmegaConf.create(
                {
                    "_target_": "TorchattacksAssessor",
                    "algorithm": "PGD",
                    "call": {
                        "target_labels": [0, 1],
                        "background_data": {"source": str(bg_dir)},
                    },
                    "visualisers": [],
                }
            )
        },
    )
    resolved = ResolvedPreprocessing(
        data_module=_ShapeModule(),
        model_module=None,
        origin="model-bundled",
        description="supplied",
    )

    monkeypatch.setattr(
        "raitap.robustness.factory.create_assessor",
        lambda _cfg: (assessor, "raitap.robustness.assessors.TorchattacksAssessor"),
    )
    monkeypatch.setattr("raitap.robustness.factory.create_robustness_visualisers", lambda _cfg: [])
    monkeypatch.setattr(
        "raitap.configs.adapter_factory.resolve_preprocessing",
        MagicMock(side_effect=AssertionError("should not resolve again")),
    )

    model = SimpleNamespace(backend=_BackendStub(torch.nn.Identity()))
    RobustnessAssessment(
        config,  # type: ignore[arg-type]
        "pgd",
        model=model,
        inputs=torch.zeros(2, 3, 8, 8),
        targets=torch.tensor([0, 1]),
        resolved_preprocessing=resolved,
    )

    bg_tensor = assessor.last_assess_kwargs["background_data"]
    assert isinstance(bg_tensor, torch.Tensor)
    assert bg_tensor.shape == (2, 3, 5, 5)
