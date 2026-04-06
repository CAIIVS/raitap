from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from raitap.models import Model
from raitap.models.backend import OnnxBackend, TorchBackend
from raitap.models.runtime import resolve_onnx_providers, resolve_torch_device

if TYPE_CHECKING:
    from pathlib import Path

    from raitap.configs.schema import AppConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeXpuModule:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _make_config(source: str, *, hardware: str = "gpu") -> AppConfig:
    return cast(
        "AppConfig",
        type(
            "AppConfig",
            (),
            {
                "model": type("ModelConfig", (), {"source": source})(),
                "hardware": hardware,
            },
        )(),
    )


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
        cfg = _make_config(str(saved_pth))
        model = Model(cfg)
        assert isinstance(model.backend, TorchBackend)
        assert isinstance(model.backend.as_model_for_explanation(), nn.Module)

    def test_loads_pt_file(self, saved_pt: Path) -> None:
        cfg = _make_config(str(saved_pt))
        model = Model(cfg)
        assert isinstance(model.backend, TorchBackend)
        assert isinstance(model.backend.as_model_for_explanation(), nn.Module)

    def test_returns_eval_mode(self, saved_pth: Path) -> None:
        cfg = _make_config(str(saved_pth))
        model = Model(cfg).backend.as_model_for_explanation()
        assert not model.training

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        cfg = _make_config(str(tmp_path / "ghost.pth"))
        with pytest.raises(FileNotFoundError, match="not found"):
            Model(cfg)

    def test_unsupported_extension_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "model.xyz"
        bad.touch()
        cfg = _make_config(str(bad))
        with pytest.raises(ValueError, match="Unsupported model format"):
            Model(cfg)

    def test_state_dict_raises_value_error(self, saved_state_dict: Path) -> None:
        cfg = _make_config(str(saved_state_dict))
        with pytest.raises(ValueError, match="state-dict"):
            Model(cfg)

    def test_non_module_object_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "tensor.pth"
        torch.save(torch.randn(3, 3), path)
        cfg = _make_config(str(path))
        with pytest.raises(ValueError, match=r"Expected an nn\.Module"):
            Model(cfg)

    def test_loads_onnx_file(self, saved_onnx: Path) -> None:
        cfg = _make_config(str(saved_onnx))
        backend = Model(cfg).backend
        assert isinstance(backend, OnnxBackend)

    def test_onnx_backend_runs_forward(self, saved_onnx: Path) -> None:
        cfg = _make_config(str(saved_onnx))
        outputs = Model(cfg).backend(torch.randn(2, 4))
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (2, 2)

    def test_invalid_hardware_raises_value_error(self, saved_pth: Path) -> None:
        cfg = _make_config(str(saved_pth), hardware="tpu")

        with pytest.raises(ValueError, match="Invalid hardware"):
            Model(cfg)

    def test_torch_hardware_cpu_sets_cpu_device(self, saved_pth: Path) -> None:
        cfg = _make_config(str(saved_pth), hardware="cpu")

        model = Model(cfg)

        assert isinstance(model.backend, TorchBackend)
        assert model.backend.device == torch.device("cpu")
        assert model.backend.hardware_label == "CPU"

    def test_torch_gpu_falls_back_to_cpu_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch, "xpu", _FakeXpuModule(False), raising=False)

        with caplog.at_level("WARNING"):
            device = resolve_torch_device("gpu")

        assert device == torch.device("cpu")
        assert "neither CUDA nor Intel XPU is available" in caplog.text

    def test_torch_gpu_selects_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch, "xpu", _FakeXpuModule(True), raising=False)

        device = resolve_torch_device("gpu")

        assert device == torch.device("cuda")

    def test_torch_gpu_selects_xpu_when_cuda_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch, "xpu", _FakeXpuModule(True), raising=False)

        device = resolve_torch_device("gpu")

        assert device == torch.device("xpu")

    def test_onnx_backend_cpu_mode_exposes_cpu_provider(self, saved_onnx: Path) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        backend = OnnxBackend.from_path(saved_onnx, hardware="cpu")

        assert backend.providers == ["CPUExecutionProvider"]

    def test_onnx_provider_resolution_prefers_cuda_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        providers = resolve_onnx_providers("gpu")

        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_onnx_provider_resolution_falls_back_to_cpu_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

        with caplog.at_level("WARNING"):
            providers = resolve_onnx_providers("gpu")

        assert providers == ["CPUExecutionProvider"]
        assert (
            "neither CUDAExecutionProvider nor OpenVINOExecutionProvider is available"
            in caplog.text
        )

    def test_onnx_provider_resolution_uses_cpu_in_cpu_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        providers = resolve_onnx_providers("cpu")

        assert providers == ["CPUExecutionProvider"]

    def test_onnx_provider_resolution_selects_openvino_when_cuda_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        )

        providers = resolve_onnx_providers("gpu")

        assert providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    def test_torch_backend_exposes_intel_xpu_hardware_label(
        self,
        monkeypatch: pytest.MonkeyPatch,
        saved_pth: Path,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch, "xpu", _FakeXpuModule(True), raising=False)

        model = Model(_make_config(str(saved_pth)))

        assert isinstance(model.backend, TorchBackend)
        assert model.backend.hardware_label == "Intel XPU"

    def test_onnx_backend_exposes_openvino_hardware_label(
        self,
        monkeypatch: pytest.MonkeyPatch,
        saved_onnx: Path,
    ) -> None:
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        )

        backend = OnnxBackend.from_path(saved_onnx, hardware="gpu")

        assert backend.hardware_label == "Intel OpenVINO"


class TestLoadModelFromName:
    def test_loads_known_model(self) -> None:
        # resnet18 is small enough to keep the test fast
        cfg = _make_config("resnet18")
        model = Model(cfg)
        assert isinstance(model.backend, TorchBackend)
        assert isinstance(model.backend.as_model_for_explanation(), nn.Module)

    def test_returns_eval_mode(self) -> None:
        cfg = _make_config("resnet18")
        model = Model(cfg).backend.as_model_for_explanation()
        assert not model.training

    def test_unknown_name_raises_value_error(self) -> None:
        cfg = _make_config("not_a_real_model_xyz")
        with pytest.raises(ValueError, match="neither an existing path nor a known"):
            Model(cfg)


class TestModelLog:
    def test_log_calls_tracker_log_model(self) -> None:
        cfg = _make_config("resnet18")
        model = Model(cfg)

        tracker = MagicMock()
        model.log(tracker)

        tracker.log_model.assert_called_once_with(model.backend)

    def test_log_passes_network_to_tracker(self) -> None:
        cfg = _make_config("resnet18")
        model = Model(cfg)

        tracker = MagicMock()
        model.log(tracker)

        logged_model = tracker.log_model.call_args[0][0]
        assert logged_model is model.backend
