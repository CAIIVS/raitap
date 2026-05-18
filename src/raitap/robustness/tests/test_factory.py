from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from raitap.robustness.assessors import TorchattacksAssessor
from raitap.robustness.contracts import MethodKind
from raitap.robustness.exceptions import MethodKindVisualiserIncompatibilityError
from raitap.robustness.factory import (
    RobustnessAssessment,
    _parse_assessor_config,
    _resolve_call_data_sources,
    check_assessor_visualiser_compat,
)
from raitap.robustness.results import ConfiguredRobustnessVisualiser
from raitap.robustness.visualisers.base_visualiser import BaseRobustnessVisualiser
from raitap.utils.errors import SampleNamesLengthError


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


# ---------------------------------------------------------------------------
# Helpers for sample_names validation tests (C1 / C2)
# ---------------------------------------------------------------------------


from raitap.models.backend import ModelBackend  # noqa: E402


class _BackendStub(ModelBackend):
    """Minimal ModelBackend stub — passes isinstance check in _require_model_backend."""

    supports_torch_autograd = True

    @property
    def hardware_label(self) -> str:
        return "stub"

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def as_model_for_explanation(self) -> torch.nn.Module:
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
    """Build a config whose raitap.sample_names is set in the config (YAML source)."""
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


# ---------------------------------------------------------------------------
# C1 / C2 factory sample_names validation tests
# ---------------------------------------------------------------------------


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
