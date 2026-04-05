from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from raitap.models import Model
from raitap.models.backend import OnnxBackend, TorchBackend

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

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


@pytest.fixture
def saved_onnx(tmp_path: Path) -> Path:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    path = tmp_path / "linear.onnx"
    weight = torch.full((4, 2), 0.25, dtype=torch.float32).numpy()
    bias = torch.tensor([0.1, -0.1], dtype=torch.float32).numpy()

    graph = helper.make_graph(
        [helper.make_node("Gemm", ["input", "weight", "bias"], ["output"])],
        "linear_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


class TestLoadModelFromPath:
    def test_loads_pth_file(self, saved_pth: Path) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": str(saved_pth)})})(),
        )
        model = Model(cfg)
        assert isinstance(model.backend, TorchBackend)
        assert isinstance(model.backend.as_model_for_explanation(), nn.Module)

    def test_loads_pt_file(self, saved_pt: Path) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": str(saved_pt)})})(),
        )
        model = Model(cfg)
        assert isinstance(model.backend, TorchBackend)
        assert isinstance(model.backend.as_model_for_explanation(), nn.Module)

    def test_returns_eval_mode(self, saved_pth: Path) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": str(saved_pth)})})(),
        )
        model = Model(cfg).backend.as_model_for_explanation()
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

    def test_loads_onnx_file(self, saved_onnx: Path) -> None:
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"model": type("ModelConfig", (), {"source": str(saved_onnx)})},
            )(),
        )
        backend = Model(cfg).backend
        assert isinstance(backend, OnnxBackend)

    def test_onnx_backend_runs_forward(self, saved_onnx: Path) -> None:
        cfg = cast(
            "AppConfig",
            type(
                "AppConfig",
                (),
                {"model": type("ModelConfig", (), {"source": str(saved_onnx)})},
            )(),
        )
        outputs = Model(cfg).backend(torch.randn(2, 4))
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (2, 2)


class TestLoadModelFromName:
    def test_loads_known_model(self) -> None:
        # resnet18 is small enough to keep the test fast
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": "resnet18"})})(),
        )
        model = Model(cfg)
        assert isinstance(model.backend, TorchBackend)
        assert isinstance(model.backend.as_model_for_explanation(), nn.Module)

    def test_returns_eval_mode(self) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": "resnet18"})})(),
        )
        model = Model(cfg).backend.as_model_for_explanation()
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

        tracker.log_model.assert_called_once_with(model.backend)

    def test_log_passes_network_to_tracker(self) -> None:
        cfg = cast(
            "AppConfig",
            type("AppConfig", (), {"model": type("ModelConfig", (), {"source": "resnet18"})})(),
        )
        model = Model(cfg)

        tracker = MagicMock()
        model.log(tracker)

        logged_model = tracker.log_model.call_args[0][0]
        assert logged_model is model.backend
