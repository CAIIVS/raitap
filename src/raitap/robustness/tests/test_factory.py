from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from raitap.configs.adapter_factory import resolve_call_data_sources
from raitap.models.base_backend import ModelBackend
from raitap.robustness.assessors import TorchattacksAssessor
from raitap.robustness.contracts import AssessmentKind
from raitap.robustness.exceptions import AssessmentKindVisualiserIncompatibilityError
from raitap.robustness.factory import (
    RobustnessAssessment,
    _parse_assessor_config,
    check_assessor_visualiser_compat,
)
from raitap.robustness.results import ConfiguredRobustnessVisualiser, RobustnessResult
from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser
from raitap.types import Capability
from raitap.utils.errors import SampleNamesLengthError

if TYPE_CHECKING:
    from pathlib import Path


class _BackendStub(ModelBackend):
    """Minimal ModelBackend stub used by factory tests."""

    provides = frozenset({Capability.AUTOGRAD})

    def __init__(self, model: torch.nn.Module | None = None) -> None:
        self._model = torch.nn.Identity() if model is None else model

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


class _OnlyFormalVisualiser(BaseRobustnessVisualiser):
    supported_assessment_kinds = frozenset({AssessmentKind.FORMAL_VERIFICATION})

    def visualise(self, result, *, context, **kwargs) -> Any:  # noqa: ANN001
        raise NotImplementedError


def _make_model_stub() -> Any:
    """Return an object whose .backend satisfies _require_model_backend."""
    return SimpleNamespace(backend=_BackendStub())


def _make_minimal_config(*, visualisers: list[Any] | None = None, tmp_path: Any = None) -> Any:
    """Build a minimal AppConfig-like namespace for PGD with no visualisers."""
    return SimpleNamespace(
        experiment_name="test_sample_names",
        _output_root=str(tmp_path) if tmp_path else "/tmp",
        transparency={},
        robustness={
            "pgd": OmegaConf.create(
                {
                    "_target_": "raitap.robustness.TorchattacksAssessor",
                    "algorithm": "PGD",
                    "constructor": {"eps": 0.03, "alpha": 0.01, "steps": 1},
                    "call": {},
                    "visualisers": visualisers or [],
                }
            )
        },
    )


def _make_yaml_names_config(*, sample_names: list[str], tmp_path: Any = None) -> Any:
    """Build a config whose raitap.sample_names is set in the config."""
    return SimpleNamespace(
        experiment_name="test_yaml_sample_names",
        _output_root=str(tmp_path) if tmp_path else "/tmp",
        transparency={},
        robustness={
            "pgd": OmegaConf.create(
                {
                    "_target_": "raitap.robustness.TorchattacksAssessor",
                    "algorithm": "PGD",
                    "constructor": {"eps": 0.03, "alpha": 0.01, "steps": 1},
                    "call": {},
                    "raitap": {"sample_names": sample_names},
                    "visualisers": [],
                }
            )
        },
    )


def test_parse_validates_top_level_keys() -> None:
    with pytest.raises(ValueError, match="Unknown robustness assessor config keys"):
        _parse_assessor_config(
            {
                "_target_": "TorchattacksAssessor",
                "algorithm": "PGD",
                "wibble": True,  # unknown key
            }
        )


def test_parse_rejects_misplaced_raitap_keys() -> None:
    # batch_size is RAITAP-owned; placing it under `call:` is a hard error.
    with pytest.raises(ValueError) as excinfo:
        _parse_assessor_config(
            {
                "_target_": "TorchattacksAssessor",
                "algorithm": "PGD",
                "call": {"eps": 0.03, "batch_size": 8},
            }
        )
    text = str(excinfo.value)
    assert "RAITAP-owned keys under 'call:'" in text
    assert "batch_size" in text


def test_resolve_call_data_sources_passes_through_non_source_dicts() -> None:
    out = resolve_call_data_sources({"target_labels": [0, 1]}, log_label="robustness call")
    assert out == {"target_labels": [0, 1]}


def test_check_visualiser_compat_raises_on_assessment_kind_mismatch() -> None:
    assessor = TorchattacksAssessor(algorithm="PGD")  # EMPIRICAL_ATTACK
    visualiser = _OnlyFormalVisualiser()
    configured = [ConfiguredRobustnessVisualiser(visualiser=visualiser)]
    with pytest.raises(AssessmentKindVisualiserIncompatibilityError):
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
        assessment_kind = AssessmentKind.EMPIRICAL_ATTACK

        def __init__(self) -> None:
            self.last_assess_kwargs: dict[str, Any] = {}

        def check_backend_compat(self, backend: object) -> None:
            del backend

        def required_capabilities(self) -> frozenset[Capability]:
            # PGD is a gradient-based empirical attack -> needs the live nn.Module,
            # so the resolver routes to autograd_module().
            return frozenset({Capability.AUTOGRAD})

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
        data_origin="model-bundled",
        model_origin="off",
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
        model=model,  # type: ignore[arg-type]
        inputs=torch.zeros(2, 3, 8, 8),
        targets=torch.tensor([0, 1]),
        resolved_preprocessing=resolved,
    )

    bg_tensor = assessor.last_assess_kwargs["background_data"]
    assert isinstance(bg_tensor, torch.Tensor)
    assert bg_tensor.shape == (2, 3, 5, 5)


def test_robustness_raises_when_runtime_sample_names_longer_than_batch(
    tmp_path: Any,
) -> None:
    config = _make_minimal_config(tmp_path=tmp_path)
    model = _make_model_stub()
    inputs = torch.zeros(2, 3, 8, 8)
    targets = torch.zeros(2, dtype=torch.long)
    with pytest.raises(SampleNamesLengthError) as info:
        RobustnessAssessment(
            config,
            "pgd",
            model,
            inputs,
            targets,
            sample_names=["a", "b", "c"],
        )
    assert info.value.got == 3
    assert info.value.expected == 2
    assert "runtime kwarg" in str(info.value)


def test_robustness_raises_when_runtime_sample_names_shorter_than_batch(
    tmp_path: Any,
) -> None:
    config = _make_minimal_config(tmp_path=tmp_path)
    model = _make_model_stub()
    inputs = torch.zeros(3, 3, 8, 8)
    targets = torch.zeros(3, dtype=torch.long)
    with pytest.raises(SampleNamesLengthError):
        RobustnessAssessment(
            config,
            "pgd",
            model,
            inputs,
            targets,
            sample_names=["only-one"],
        )


def test_robustness_raises_when_yaml_sample_names_mismatch(
    tmp_path: Any,
) -> None:
    config = _make_yaml_names_config(sample_names=["x", "y"], tmp_path=tmp_path)
    model = _make_model_stub()
    inputs = torch.zeros(3, 3, 8, 8)
    targets = torch.zeros(3, dtype=torch.long)
    with pytest.raises(SampleNamesLengthError) as info:
        RobustnessAssessment(
            config,
            "pgd",
            model,
            inputs,
            targets,
        )
    assert "raitap.sample_names" in str(info.value)


def test_robustness_does_not_raise_when_sample_names_is_none(
    tmp_path: Any,
) -> None:
    config = _make_minimal_config(tmp_path=tmp_path)
    model = _make_model_stub()
    inputs = torch.zeros(2, 3, 8, 8)
    targets = torch.zeros(2, dtype=torch.long)
    try:
        RobustnessAssessment(
            config,
            "pgd",
            model,
            inputs,
            targets,
            sample_names=None,
        )
    except SampleNamesLengthError:
        pytest.fail("SampleNamesLengthError raised unexpectedly for sample_names=None")
    except Exception:
        pass  # other errors (e.g. from assessor) are out of scope


def test_robustness_does_not_raise_when_sample_names_length_matches(
    tmp_path: Any,
) -> None:
    config = _make_minimal_config(tmp_path=tmp_path)
    model = _make_model_stub()
    inputs = torch.zeros(2, 3, 8, 8)
    targets = torch.zeros(2, dtype=torch.long)
    try:
        RobustnessAssessment(
            config,
            "pgd",
            model,
            inputs,
            targets,
            sample_names=["a", "b"],
        )
    except SampleNamesLengthError:
        pytest.fail("SampleNamesLengthError raised unexpectedly for matching sample_names")
    except Exception:
        pass  # other errors (e.g. from assessor) are out of scope
