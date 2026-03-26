from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from raitap.models import Model
from raitap.models.converters import CONVERTERS, FormatConverter, PthConverter

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


class TestConvertersRegistry:
    def test_pth_and_pt_registered(self) -> None:
        assert ".pth" in CONVERTERS
        assert ".pt" in CONVERTERS

    def test_all_converters_satisfy_protocol(self) -> None:
        for converter in CONVERTERS.values():
            assert isinstance(converter, FormatConverter)


class TestPthConverter:
    def test_returns_model(self, tmp_path: Path) -> None:
        model = nn.Sequential(nn.Linear(2, 1))
        model.eval()
        p = tmp_path / "model.pth"
        torch.save(model, p)
        result = PthConverter().convert(p)
        assert isinstance(result, nn.Module)
        assert not result.training


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model() -> nn.Module:
    model = nn.Sequential(nn.Linear(4, 2))
    model.eval()
    return model


@pytest.fixture
def saved_pth(tmp_path: Path, tiny_model: nn.Module) -> Path:
    path = tmp_path / "model.pth"
    torch.save(tiny_model, path)
    return path


@pytest.fixture
def saved_pt(tmp_path: Path, tiny_model: nn.Module) -> Path:
    path = tmp_path / "model.pt"
    torch.save(tiny_model, path)
    return path


@pytest.fixture
def saved_state_dict(tmp_path: Path, tiny_model: nn.Module) -> Path:
    path = tmp_path / "state_dict.pth"
    torch.save(tiny_model.state_dict(), path)
    return path


class TestLoadModelFromPath:
    def test_loads_pth_file(self, saved_pth: Path) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": str(saved_pth)})})(),
        )
        model = Model(cfg).network
        assert isinstance(model, nn.Module)

    def test_loads_pt_file(self, saved_pt: Path) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": str(saved_pt)})})(),
        )
        model = Model(cfg).network
        assert isinstance(model, nn.Module)

    def test_returns_eval_mode(self, saved_pth: Path) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": str(saved_pth)})})(),
        )
        model = Model(cfg).network
        assert not model.training

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"model": type("ModelConfig", (), {"source": str(tmp_path / "ghost.pth")})},
            )(),
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            Model(cfg)

    def test_unsupported_extension_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "model.xyz"
        bad.touch()
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": str(bad)})})(),
        )
        with pytest.raises(ValueError, match="Unsupported model format"):
            Model(cfg)

    def test_state_dict_raises_value_error(self, saved_state_dict: Path) -> None:
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"model": type("ModelConfig", (), {"source": str(saved_state_dict)})},
            )(),
        )
        with pytest.raises(ValueError, match="state-dict"):
            Model(cfg)

    def test_non_module_object_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "tensor.pth"
        torch.save(torch.randn(3, 3), path)
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": str(path)})})(),
        )
        with pytest.raises(ValueError, match=r"Expected an nn\.Module"):
            Model(cfg)


class TestLoadModelFromName:
    def test_loads_known_model(self) -> None:
        # resnet18 is small enough to keep the test fast
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": "resnet18"})})(),
        )
        model = Model(cfg).network
        assert isinstance(model, nn.Module)

    def test_returns_eval_mode(self) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": "resnet18"})})(),
        )
        model = Model(cfg).network
        assert not model.training

    def test_unknown_name_raises_value_error(self) -> None:
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"model": type("ModelConfig", (), {"source": "not_a_real_model_xyz"})},
            )(),
        )
        with pytest.raises(ValueError, match="neither an existing path nor a known"):
            Model(cfg)


class TestModelLog:
    def test_log_calls_tracker_log_model(self) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": "resnet18"})})(),
        )
        model = Model(cfg)

        tracker = MagicMock()
        model.log(tracker)

        tracker.log_model.assert_called_once_with(model.network)

    def test_log_passes_network_to_tracker(self) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": "resnet18"})})(),
        )
        model = Model(cfg)

        tracker = MagicMock()
        model.log(tracker)

        logged_model = tracker.log_model.call_args[0][0]
        assert logged_model is model.network
        assert isinstance(logged_model, torch.nn.Module)
