from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from raitap.models import load_model
from raitap.models.converters import CONVERTERS, FormatConverter, PthConverter

# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


class TestConvertersRegistry:
    def test_pth_and_pt_registered(self):
        assert ".pth" in CONVERTERS
        assert ".pt" in CONVERTERS

    def test_all_converters_satisfy_protocol(self):
        for converter in CONVERTERS.values():
            assert isinstance(converter, FormatConverter)


class TestPthConverter:
    def test_returns_same_path(self, tmp_path: Path):
        p = tmp_path / "model.pth"
        p.touch()
        assert PthConverter().convert(p) == p


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
    def test_loads_pth_file(self, saved_pth: Path):
        model = load_model(saved_pth)
        assert isinstance(model, nn.Module)

    def test_loads_pt_file(self, saved_pt: Path):
        model = load_model(saved_pt)
        assert isinstance(model, nn.Module)

    def test_accepts_string_path(self, saved_pth: Path):
        model = load_model(str(saved_pth))
        assert isinstance(model, nn.Module)

    def test_returns_eval_mode(self, saved_pth: Path):
        model = load_model(saved_pth)
        assert not model.training

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_model(tmp_path / "ghost.pth")

    def test_unsupported_extension_raises_value_error(self, tmp_path: Path):
        bad = tmp_path / "model.xyz"
        bad.touch()
        with pytest.raises(ValueError, match="Unsupported model format"):
            load_model(bad)

    def test_state_dict_raises_value_error(self, saved_state_dict: Path):
        with pytest.raises(ValueError, match="state-dict"):
            load_model(saved_state_dict)

    def test_non_module_object_raises_value_error(self, tmp_path: Path):
        path = tmp_path / "tensor.pth"
        torch.save(torch.randn(3, 3), path)
        with pytest.raises(ValueError, match=r"Expected an nn\.Module"):
            load_model(path)


class TestLoadModelFromName:
    def test_loads_known_model(self):
        # resnet18 is small enough to keep the test fast
        model = load_model("resnet18")
        assert isinstance(model, nn.Module)

    def test_returns_eval_mode(self):
        model = load_model("resnet18")
        assert not model.training

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="neither an existing path nor a known"):
            load_model("not_a_real_model_xyz")
