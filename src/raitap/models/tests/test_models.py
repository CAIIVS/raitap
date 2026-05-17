from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from raitap.models import Model, runtime
from raitap.models.backend import (
    OnnxBackend,
    TorchBackend,
    _adapt_input_shape,
    _resolve_onnx_expected_shape,
)
from raitap.models.runtime import resolve_onnx_providers, resolve_torch_device
from raitap.types import Hardware
from raitap.utils.errors import ModelInputShapeError

if TYPE_CHECKING:
    from pathlib import Path

    import onnxruntime as ort

    from raitap.configs.schema import AppConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeXpuModule:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeOnnxValueInfo:
    def __init__(
        self,
        name: str,
        type_name: str = "tensor(float)",
        shape: list[int | str | None] | None = None,
    ) -> None:
        self.name = name
        self.type = type_name
        self.shape = shape if shape is not None else ["batch", 4]


class _FakeOnnxSession:
    def get_inputs(self) -> list[_FakeOnnxValueInfo]:
        return [_FakeOnnxValueInfo("input")]

    def get_outputs(self) -> list[_FakeOnnxValueInfo]:
        return [_FakeOnnxValueInfo("output")]


def _as_inference_session(session: _FakeOnnxSession) -> ort.InferenceSession:
    return cast("ort.InferenceSession", session)


def _make_config(
    source: str,
    *,
    hardware: str = "gpu",
    arch: str | None = None,
    num_classes: int | None = None,
    pretrained: bool = False,
    allow_unsafe_pickle: bool = False,
) -> AppConfig:
    return cast(
        "AppConfig",
        type(
            "AppConfig",
            (),
            {
                "model": type(
                    "ModelConfig",
                    (),
                    {
                        "source": source,
                        "arch": arch,
                        "num_classes": num_classes,
                        "pretrained": pretrained,
                        "allow_unsafe_pickle": allow_unsafe_pickle,
                    },
                )(),
                "data": type(
                    "DataConfig",
                    (),
                    {
                        "preprocessing": None,
                        "acknowledge_preprocessing_off": True,
                        "acknowledge_preprocessing_exec": False,
                        "input_metadata": None,
                    },
                )(),
                "hardware": hardware,
            },
        )(),
    )


def _cuda_not_available() -> bool:
    return not torch.cuda.is_available()


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
        cfg = _make_config(str(saved_pth), allow_unsafe_pickle=True)
        with pytest.warns(DeprecationWarning, match="pickled nn.Module"):
            model = Model(cfg)
        assert isinstance(model.backend, TorchBackend)
        assert isinstance(model.backend.as_model_for_explanation(), nn.Module)

    def test_loads_pt_file(self, saved_pt: Path) -> None:
        cfg = _make_config(str(saved_pt), allow_unsafe_pickle=True)
        with pytest.warns(DeprecationWarning, match="pickled nn.Module"):
            model = Model(cfg)
        assert isinstance(model.backend, TorchBackend)
        assert isinstance(model.backend.as_model_for_explanation(), nn.Module)

    def test_returns_eval_mode(self, saved_pth: Path) -> None:
        cfg = _make_config(str(saved_pth), allow_unsafe_pickle=True)
        with pytest.warns(DeprecationWarning, match="pickled nn.Module"):
            model = Model(cfg).backend.as_model_for_explanation()
        assert not model.training

    def test_pickled_module_load_refused_without_opt_in(self, saved_pth: Path) -> None:
        cfg = _make_config(str(saved_pth))
        with pytest.raises(ValueError, match=r"Refusing to load.*allow_unsafe_pickle"):
            Model(cfg)

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

    def test_state_dict_loads_with_arch_and_num_classes(self, tmp_path: Path) -> None:
        from torchvision import models as tv_models

        ref = tv_models.resnet18(weights=None, num_classes=3)
        path = tmp_path / "weights.pth"
        torch.save(ref.state_dict(), path)
        cfg = _make_config(str(path), arch="resnet18", num_classes=3)
        model = Model(cfg)
        assert isinstance(model.backend, TorchBackend)
        loaded = model.backend.as_model_for_explanation()
        # Round-trip: identical parameter values.
        for k, v in ref.state_dict().items():
            assert torch.equal(v, loaded.state_dict()[k].cpu())

    def test_state_dict_without_arch_raises(self, saved_state_dict: Path) -> None:
        cfg = _make_config(str(saved_state_dict))
        with pytest.raises(ValueError, match="State-dict loading requires"):
            Model(cfg)

    def test_state_dict_with_mismatched_num_classes_raises(self, tmp_path: Path) -> None:
        from torchvision import models as tv_models

        ref = tv_models.resnet18(weights=None, num_classes=10)
        path = tmp_path / "weights.pth"
        torch.save(ref.state_dict(), path)
        cfg = _make_config(str(path), arch="resnet18", num_classes=2)
        with pytest.raises(RuntimeError, match=r"size mismatch|shape"):
            Model(cfg)

    def test_torchscript_load_via_jit(self, tmp_path: Path) -> None:
        ref = nn.Linear(4, 2).eval()
        scripted = torch.jit.script(ref)
        path = tmp_path / "scripted.pt"
        scripted.save(str(path))
        cfg = _make_config(str(path), hardware="cpu")
        loaded = Model(cfg).backend.as_model_for_explanation()
        assert isinstance(loaded, torch.jit.ScriptModule)
        x = torch.randn(2, 4)
        torch.testing.assert_close(loaded(x), ref(x))

    def test_pickled_module_load_emits_deprecation_warning(self, saved_pth: Path) -> None:
        cfg = _make_config(str(saved_pth), allow_unsafe_pickle=True)
        with pytest.warns(
            DeprecationWarning,
            match=r"Loading pickled nn\.Module.*Prefer.*state_dict",
        ):
            Model(cfg)

    def test_non_module_object_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "tensor.pth"
        torch.save(torch.randn(3, 3), path)
        cfg = _make_config(str(path))
        with pytest.raises(ValueError, match=r"Expected an nn\.Module or state-dict"):
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

    def test_onnx_forward_numpy_matches_call(self, saved_onnx: Path) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        backend = OnnxBackend.from_path(saved_onnx, hardware="cpu")
        x_np = np.random.randn(3, 4).astype(np.float32)
        tensor_out = backend(torch.from_numpy(x_np))
        numpy_out = backend.forward_numpy(x_np)
        np.testing.assert_allclose(
            tensor_out.detach().cpu().numpy(), numpy_out, rtol=1e-5, atol=1e-6
        )

    def test_invalid_hardware_raises_value_error(self, saved_pth: Path) -> None:
        cfg = _make_config(str(saved_pth), hardware="tpu")

        with pytest.raises(ValueError, match="Invalid hardware"):
            Model(cfg)

    def test_torch_hardware_cpu_sets_cpu_device(self, saved_pth: Path) -> None:
        cfg = _make_config(str(saved_pth), hardware="cpu", allow_unsafe_pickle=True)

        with pytest.warns(DeprecationWarning, match="pickled nn.Module"):
            model = Model(cfg)

        assert isinstance(model.backend, TorchBackend)
        assert model.backend.device == torch.device("cpu")
        assert model.backend.hardware_label == "CPU"

    @pytest.mark.runtime
    def test_torch_gpu_falls_back_to_cpu_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(runtime, "_torch_mps_is_available", lambda: False)
        monkeypatch.setattr(runtime, "_torch_xpu_is_available", lambda: False)

        with pytest.warns(UserWarning, match="neither CUDA nor Intel XPU is available"):
            device = resolve_torch_device("gpu")

        assert device == torch.device("cpu")

    @pytest.mark.runtime
    def test_torch_gpu_selects_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(runtime, "_torch_mps_is_available", lambda: True)
        monkeypatch.setattr(runtime, "_torch_xpu_is_available", lambda: True)

        device = resolve_torch_device("gpu")

        assert device == torch.device("cuda")

    @pytest.mark.runtime
    def test_torch_gpu_falls_back_to_cpu_when_mps_is_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(runtime, "_torch_mps_is_available", lambda: True)
        monkeypatch.setattr(runtime, "_torch_xpu_is_available", lambda: False)

        with pytest.warns(UserWarning, match="Apple MPS support is temporarily disabled"):
            device = resolve_torch_device("gpu")

        assert device == torch.device("cpu")

    @pytest.mark.runtime
    def test_torch_gpu_prefers_cuda_over_mps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(runtime, "_torch_mps_is_available", lambda: True)
        monkeypatch.setattr(runtime, "_torch_xpu_is_available", lambda: False)

        device = resolve_torch_device("gpu")

        assert device == torch.device("cuda")

    @pytest.mark.runtime
    def test_torch_gpu_disables_apple_mps_even_when_it_is_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(runtime, "_torch_mps_is_available", lambda: True)
        monkeypatch.setattr(runtime, "_torch_xpu_is_available", lambda: True)

        device = resolve_torch_device("gpu")

        assert device == torch.device("cpu")

    @pytest.mark.runtime
    def test_torch_gpu_selects_xpu_when_cuda_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(runtime, "_torch_mps_is_available", lambda: False)
        monkeypatch.setattr(runtime, "_torch_xpu_is_available", lambda: True)

        device = resolve_torch_device("gpu")

        assert device == torch.device("xpu")

    @pytest.mark.runtime
    def test_onnx_backend_cpu_mode_exposes_cpu_provider(self, saved_onnx: Path) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        backend = OnnxBackend.from_path(saved_onnx, hardware="cpu")

        assert backend.providers == ["CPUExecutionProvider"]

    @pytest.mark.runtime
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

    @pytest.mark.runtime
    def test_onnx_provider_resolution_falls_back_to_cpu_when_coreml_is_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )

        with pytest.warns(UserWarning, match="Apple CoreML support is temporarily disabled"):
            providers = resolve_onnx_providers("gpu")

        assert providers == ["CPUExecutionProvider"]

    @pytest.mark.runtime
    def test_onnx_provider_resolution_prefers_cuda_over_disabled_apple_coreml(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        providers = resolve_onnx_providers("gpu")

        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @pytest.mark.runtime
    def test_onnx_provider_resolution_prefers_openvino_over_disabled_apple_coreml(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: [
                "OpenVINOExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        providers = resolve_onnx_providers("gpu")

        assert providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    @pytest.mark.runtime
    def test_onnx_provider_resolution_falls_back_to_cpu_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

        with pytest.warns(
            UserWarning,
            match="neither CUDAExecutionProvider nor OpenVINOExecutionProvider is available",
        ):
            providers = resolve_onnx_providers("gpu")

        assert providers == ["CPUExecutionProvider"]

    @pytest.mark.runtime
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

    @pytest.mark.runtime
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

    @pytest.mark.runtime
    def test_torch_backend_exposes_intel_xpu_hardware_label(
        self,
    ) -> None:
        backend = TorchBackend(nn.Identity(), device=torch.device("xpu"))

        assert backend.hardware_label == "Intel XPU"

    @pytest.mark.runtime
    def test_onnx_backend_exposes_openvino_hardware_label(self) -> None:
        backend = OnnxBackend(
            _as_inference_session(_FakeOnnxSession()),
            providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        )

        assert backend.hardware_label == "Intel OpenVINO"

    @pytest.mark.cuda
    @pytest.mark.skipif(_cuda_not_available(), reason="CUDA is not available")
    def test_torch_gpu_uses_real_cuda_backend(self, saved_pth: Path) -> None:
        cfg = _make_config(str(saved_pth), hardware="gpu", allow_unsafe_pickle=True)

        with pytest.warns(DeprecationWarning, match="pickled nn.Module"):
            model = Model(cfg)

        assert isinstance(model.backend, TorchBackend)
        assert model.backend.device.type == "cuda"

        prepared_inputs = model.backend._prepare_inputs(torch.randn(2, 4))
        assert prepared_inputs.device.type == "cuda"

        outputs = model.backend(prepared_inputs)
        assert isinstance(outputs, torch.Tensor)
        assert outputs.device.type == "cuda"
        assert model.backend.hardware_label == "CUDA"

    @pytest.mark.cuda
    @pytest.mark.skipif(_cuda_not_available(), reason="CUDA is not available")
    def test_onnx_gpu_uses_cuda_execution_provider(self, saved_onnx: Path) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CUDAExecutionProvider is not available")

        backend = OnnxBackend.from_path(saved_onnx, hardware="gpu")
        outputs = backend(torch.randn(2, 4))

        assert backend.providers[0] == "CUDAExecutionProvider"
        assert backend.session.get_providers()[0] == "CUDAExecutionProvider"
        assert backend.hardware_label == "CUDA"
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (2, 2)


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


@pytest.fixture
def saved_onnx_tabular(tmp_path: Path) -> Path:
    """Tiny ONNX model declaring input shape [1, 5] (2D tabular)."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    path = tmp_path / "tabular.onnx"
    weight = torch.full((5, 2), 0.1, dtype=torch.float32).numpy()
    bias = torch.tensor([0.0, 0.0], dtype=torch.float32).numpy()

    graph = helper.make_graph(
        [helper.make_node("Gemm", ["input", "weight", "bias"], ["output"])],
        "tabular_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 5])],
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


class TestModelInputShapeAdapter:
    def test_adapt_input_shape_passthrough_when_expected_is_none(self) -> None:
        t = torch.randn(3, 4)
        result = _adapt_input_shape(t, None)
        assert result is t

    def test_adapt_input_shape_passthrough_when_already_matches(self) -> None:
        t = torch.randn(3, 4)
        result = _adapt_input_shape(t, (None, 4))
        assert result is t
        assert result.shape == (3, 4)

    def test_adapt_input_shape_reshapes_when_numel_matches_torch(self) -> None:
        t = torch.randn(7, 5)
        result = _adapt_input_shape(t, (None, 1, 1, 5))
        assert isinstance(result, torch.Tensor)
        assert result.shape == (7, 1, 1, 5)
        # Values preserved (same underlying storage via reshape).
        torch.testing.assert_close(result.reshape(7, 5), t)

    def test_adapt_input_shape_reshapes_when_numel_matches_numpy(self) -> None:
        arr = np.random.randn(7, 5).astype(np.float32)
        result = _adapt_input_shape(arr, (None, 1, 1, 5))
        assert isinstance(result, np.ndarray)
        assert result.shape == (7, 1, 1, 5)
        np.testing.assert_array_equal(result.reshape(7, 5), arr)

    def test_adapt_input_shape_raises_on_numel_mismatch(self) -> None:
        t = torch.randn(3, 6)
        with pytest.raises(ModelInputShapeError) as exc_info:
            _adapt_input_shape(t, (None, 1, 1, 5))
        msg = str(exc_info.value)
        assert "(3, 6)" in msg
        # Expected shape rendered with 'N' for the batch dim.
        assert "1, 1, 5" in msg
        assert "data.input_metadata.shape" in msg
        # Carries structured fields.
        assert exc_info.value.input_shape == (3, 6)
        assert exc_info.value.expected_shape == (None, 1, 1, 5)

    def test_resolve_onnx_expected_shape_respects_concrete_dims(self) -> None:
        # All-concrete dims pass through unchanged. The graph's fixed batch
        # dim is the source of truth — callers feeding a mismatching batch
        # get a typed error from `_adapt_input_shape`, not a cryptic ORT one.
        assert _resolve_onnx_expected_shape([1, 1, 1, 5]) == (1, 1, 1, 5)
        assert _resolve_onnx_expected_shape([1, 3, 224, 224]) == (1, 3, 224, 224)

    def test_resolve_onnx_expected_shape_marks_symbolic_dim_dynamic(self) -> None:
        # Symbolic batch dim -> `None` so `_adapt_input_shape` resolves it
        # from `inputs.shape[0]` at runtime.
        assert _resolve_onnx_expected_shape(["batch", 3, 224, 224]) == (None, 3, 224, 224)

    def test_resolve_onnx_expected_shape_raises_on_ambiguous(self) -> None:
        # Two or more symbolic / unknown dims -> no single reshape target.
        with pytest.raises(ModelInputShapeError) as exc_info:
            _resolve_onnx_expected_shape(["batch", 3, "h", "w"])
        # Ambiguous variant: input_shape is None.
        assert exc_info.value.input_shape is None
        assert "ambiguous" in str(exc_info.value).lower()
        assert "data.input_metadata.shape" in str(exc_info.value)

    def test_resolve_onnx_expected_shape_empty_returns_none(self) -> None:
        assert _resolve_onnx_expected_shape([]) is None

    def test_onnx_backend_auto_reshapes_tabular_input(self, saved_onnx_tabular: Path) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        backend = OnnxBackend.from_path(saved_onnx_tabular, hardware="cpu")
        # Backend declared [1, 5] -> resolved to (None, 5).
        assert backend.expected_input_shape == (None, 5)
        # Feed batch (N, 5) directly -- already matches.
        x = torch.randn(4, 5)
        tensor_out = backend(x)
        assert isinstance(tensor_out, torch.Tensor)
        assert tensor_out.shape == (4, 2)
        # forward_numpy path too.
        numpy_out = backend.forward_numpy(x.numpy())
        assert numpy_out.shape == (4, 2)
        np.testing.assert_allclose(
            tensor_out.detach().cpu().numpy(), numpy_out, rtol=1e-5, atol=1e-6
        )

    def test_onnx_backend_reshapes_when_caller_supplies_flat_batch(self, tmp_path: Path) -> None:
        """ONNX with [1,1,1,5] should accept (N,5) inputs via auto-reshape."""
        onnx = pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        from onnx import TensorProto, helper, numpy_helper

        path = tmp_path / "acasxu_like.onnx"
        # Flatten -> Gemm to make a [1,1,1,5] input model.
        weight = torch.full((5, 2), 0.1, dtype=torch.float32).numpy()
        bias = torch.tensor([0.0, 0.0], dtype=torch.float32).numpy()
        shape_init = np.array([-1, 5], dtype=np.int64)
        graph = helper.make_graph(
            [
                helper.make_node("Reshape", ["input", "new_shape"], ["flat"]),
                helper.make_node("Gemm", ["flat", "weight", "bias"], ["output"]),
            ],
            "acasxu_like",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 1, 1, 5])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
            [
                numpy_helper.from_array(weight, name="weight"),
                numpy_helper.from_array(bias, name="bias"),
                numpy_helper.from_array(shape_init, name="new_shape"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.checker.check_model(model)
        onnx.save(model, path)

        backend = OnnxBackend.from_path(path, hardware="cpu")
        assert backend.expected_input_shape == (None, 1, 1, 5)
        # Feed (N, 5) -- should be auto-reshaped to (N, 1, 1, 5).
        tensor_out = backend(torch.randn(3, 5))
        assert tensor_out.shape == (3, 2)
        numpy_out = backend.forward_numpy(np.random.randn(3, 5).astype(np.float32))
        assert numpy_out.shape == (3, 2)

    def test_torch_backend_honours_explicit_expected_input_shape(self) -> None:
        model = nn.Linear(5, 2).eval()
        backend = TorchBackend(model, device=torch.device("cpu"))
        backend.expected_input_shape = (None, 5)

        out = backend(backend._prepare_inputs(torch.randn(4, 5)))
        assert out.shape == (4, 2)

        with pytest.raises(ModelInputShapeError):
            backend._prepare_inputs(torch.randn(4, 4))

    def test_torch_backend_reshapes_via_prepare_inputs(self) -> None:
        model = nn.Linear(5, 2).eval()
        backend = TorchBackend(model, device=torch.device("cpu"))
        backend.expected_input_shape = (None, 1, 1, 5)

        prepared = backend._prepare_inputs(torch.randn(3, 5))
        assert prepared.shape == (3, 1, 1, 5)


class TestModelPreprocessingWrap:
    """``Model._load_model`` + ``_apply_preprocessing`` integration.

    Verifies that the preprocessing option resolved from the config is
    wrapped around the backbone the way the rest of the pipeline expects:
    off leaves the backend untouched, model-bundled / custom-file replace
    ``backend.model`` with ``nn.Sequential(preprocessing, original)``, and
    ONNX backends with an active preprocessing raise loudly instead of
    silently dropping it.
    """

    @staticmethod
    def _fixture_path() -> Path:
        from pathlib import Path as _Path

        return (
            _Path(__file__).resolve().parents[2]
            / "data"
            / "tests"
            / "fixtures"
            / "preproc_imagenet.py"
        )

    def test_off_leaves_backbone_untouched(self) -> None:
        from torchvision.models.resnet import ResNet

        from raitap.configs.schema import AppConfig, DataConfig, ModelConfig

        cfg = cast(
            "AppConfig",
            AppConfig(model=ModelConfig(source="resnet18"), data=DataConfig()),
        )
        model = Model(cfg)

        assert model.resolved_preprocessing.origin == "off"
        assert model.resolved_preprocessing.data_module is None
        assert model.resolved_preprocessing.model_module is None
        assert isinstance(model.backend, TorchBackend)
        assert not isinstance(model.backend.model, nn.Sequential)
        assert isinstance(model.backend.model, ResNet)

    def test_model_bundled_wraps_backbone_with_sequential(self) -> None:
        from torchvision.models.resnet import ResNet
        from torchvision.transforms import v2

        from raitap.configs.schema import AppConfig, DataConfig, ModelConfig

        cfg = cast(
            "AppConfig",
            AppConfig(
                model=ModelConfig(source="resnet18"),
                data=DataConfig(preprocessing="model-bundled"),
            ),
        )
        model = Model(cfg)

        assert model.resolved_preprocessing.origin == "model-bundled"
        # data_module (Resize + CenterCrop) lives in the loader, not in the
        # model wrap. The wrap only carries the value half.
        assert model.resolved_preprocessing.data_module is not None
        assert isinstance(model.resolved_preprocessing.model_module, v2.Normalize)
        assert isinstance(model.backend, TorchBackend)
        assert isinstance(model.backend.model, nn.Sequential)
        children = list(model.backend.model.children())
        assert len(children) == 2
        # child[0] is Normalize, child[1] is the original backbone.
        assert isinstance(children[0], v2.Normalize)
        assert isinstance(children[1], ResNet)

    def test_supplied_resolved_preprocessing_skips_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from raitap.configs.schema import AppConfig, DataConfig, ModelConfig
        from raitap.data.preprocessing import ResolvedPreprocessing

        cfg = cast(
            "AppConfig",
            AppConfig(
                model=ModelConfig(source="resnet18"),
                data=DataConfig(preprocessing="model-bundled"),
            ),
        )
        resolved = ResolvedPreprocessing(
            data_module=None,
            model_module=None,
            origin="model-bundled",
            description="supplied",
        )
        monkeypatch.setattr(
            "raitap.models.model.resolve_preprocessing",
            MagicMock(side_effect=AssertionError("should not resolve again")),
        )
        monkeypatch.setattr(
            Model,
            "_load_model",
            lambda _self, _config: TorchBackend(nn.Identity(), device=torch.device("cpu")),
        )

        model = Model(cfg, resolved_preprocessing=resolved)

        assert model.resolved_preprocessing is resolved

    def test_wrap_clones_model_preprocessing_before_device_mutation(self) -> None:
        from raitap.data.preprocessing import ResolvedPreprocessing
        from raitap.models.model import _apply_preprocessing

        cfg = _make_config("resnet18")
        backend = TorchBackend(nn.Identity(), device=torch.device("cpu"))
        model_module = nn.Dropout()
        assert model_module.training
        resolved = ResolvedPreprocessing(
            data_module=None,
            model_module=model_module,
            origin="custom-file",
            description="supplied",
        )

        returned = _apply_preprocessing(
            backend,
            cfg,
            resolved_preprocessing=resolved,
        )

        assert returned is resolved
        assert model_module.training
        assert isinstance(backend.model, nn.Sequential)
        wrapped_preprocessing = backend.model[0]
        assert wrapped_preprocessing is not model_module
        assert not wrapped_preprocessing.training

    def test_model_bundled_wrap_consumes_raw_input(self) -> None:
        from raitap.configs.schema import AppConfig, DataConfig, ModelConfig

        cfg = cast(
            "AppConfig",
            AppConfig(
                model=ModelConfig(source="resnet18"),
                data=DataConfig(preprocessing="model-bundled"),
            ),
        )
        model = Model(cfg)

        raw = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            out = model.backend(raw)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 1000)

    def test_custom_file_wraps_with_user_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from raitap.configs.schema import AppConfig, DataConfig, ModelConfig

        monkeypatch.delenv("RAITAP_ALLOW_PREPROCESSING_EXEC", raising=False)
        fixture = self._fixture_path()
        assert fixture.exists(), f"fixture missing: {fixture}"

        cfg = cast(
            "AppConfig",
            AppConfig(
                model=ModelConfig(source="resnet18"),
                data=DataConfig(
                    preprocessing=str(fixture),
                    acknowledge_preprocessing_exec=True,
                ),
            ),
        )
        model = Model(cfg)

        assert model.resolved_preprocessing.origin == "custom-file"
        assert isinstance(model.backend, TorchBackend)
        assert isinstance(model.backend.model, nn.Sequential)
        assert model.resolved_preprocessing.file_sha256 is not None
        assert len(model.resolved_preprocessing.file_sha256) == 64

    def test_onnx_backend_with_active_preprocessing_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from raitap.configs.schema import AppConfig, DataConfig, ModelConfig
        from raitap.models.model import _apply_preprocessing

        monkeypatch.delenv("RAITAP_ALLOW_PREPROCESSING_EXEC", raising=False)

        backend = OnnxBackend(
            _as_inference_session(_FakeOnnxSession()),
            providers=["CPUExecutionProvider"],
        )
        # arch is required so the model-bundled resolver can find weights
        # before the ONNX check fires; otherwise the test would fail on a
        # different (unrelated) error path.
        cfg = cast(
            "AppConfig",
            AppConfig(
                model=ModelConfig(source="resnet18", arch="resnet18"),
                data=DataConfig(preprocessing="model-bundled"),
            ),
        )
        with pytest.raises(NotImplementedError, match="ONNX"):
            _apply_preprocessing(backend, cfg)


def test_resnet_state_dict_with_model_bundled_preprocessing_keeps_gradcam_path(
    tmp_path: Path,
) -> None:
    from torchvision import models as tv_models
    from torchvision.models.resnet import ResNet
    from torchvision.transforms import v2

    from raitap.configs.schema import AppConfig, DataConfig, ModelConfig

    ref = tv_models.resnet50(weights=None, num_classes=7)
    path = tmp_path / "lwise_ham10000_inner_resnet50_state_dict.pt"
    torch.save(ref.state_dict(), path)
    cfg = cast(
        "AppConfig",
        AppConfig(
            model=ModelConfig(source=str(path), arch="resnet50", num_classes=7),
            data=DataConfig(preprocessing="model-bundled"),
            hardware=Hardware.cpu,
        ),
    )

    model = Model(cfg)

    assert cfg.model.allow_unsafe_pickle is False
    assert model.resolved_preprocessing.origin == "model-bundled"
    assert isinstance(model.resolved_preprocessing.model_module, v2.Normalize)
    assert isinstance(model.backend, TorchBackend)
    assert isinstance(model.backend.model, nn.Sequential)
    children = list(model.backend.model.children())
    assert len(children) == 2
    assert isinstance(children[1], ResNet)
    assert children[1].layer4[2].conv3 is not None
