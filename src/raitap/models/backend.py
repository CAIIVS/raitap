from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from .runtime import _ONNX_RUNTIME_INSTALL_HINT, resolve_onnx_providers

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import onnxruntime as ort

logger = logging.getLogger(__name__)


_NUMPY_DTYPES_BY_ONNX_TYPE: dict[str, np.dtype[Any]] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(bool)": np.dtype(np.bool_),
}


class ModelBackend(ABC):
    """Backend-agnostic model runtime interface."""

    supports_torch_autograd: bool

    @property
    @abstractmethod
    def hardware_label(self) -> str:
        """Human-readable label for the resolved runtime backend."""

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Adapt runtime inputs to this backend's preferred device/layout."""
        return inputs

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Adapt explainer/runtime kwargs for this backend."""
        return kwargs

    @abstractmethod
    def __call__(self, inputs: torch.Tensor) -> Any:
        """Run inference for ``inputs``."""

    @abstractmethod
    def as_model_for_explanation(self) -> nn.Module:
        """Return the model object that explainers should consume."""


class TorchBackend(ModelBackend):
    """PyTorch-backed model runtime."""

    supports_torch_autograd = True

    def __init__(self, model: nn.Module, *, device: torch.device | None = None) -> None:
        self.model = model
        self.device = torch.device("cpu") if device is None else device

    @property
    def hardware_label(self) -> str:
        return _torch_hardware_label(self.device)

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.to(self.device)

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return _move_tensors_to_device(kwargs, self.device)

    def __call__(self, inputs: torch.Tensor) -> Any:
        return self.model(inputs)

    def as_model_for_explanation(self) -> nn.Module:
        return self.model


class _OnnxExplanationModule(nn.Module):
    def __init__(self, backend: OnnxBackend) -> None:
        super().__init__()
        self.backend = backend

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.backend(inputs)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "OnnxBackend explanation bridge expected a torch.Tensor output, "
                f"got {type(output).__name__}."
            )
        return output


class OnnxBackend(ModelBackend):
    """ONNX Runtime-backed model runtime."""

    supports_torch_autograd = False

    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        providers: Sequence[str],
        model_path: Path | None = None,
    ) -> None:
        self.session = session
        self.providers = list(providers)
        self.model_path = model_path
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"Expected exactly one ONNX input, got {len(inputs)}. "
                "Multi-input ONNX models are not supported yet."
            )
        self.input_name = inputs[0].name
        self.input_type = inputs[0].type
        self.output_names = [output.name for output in session.get_outputs()]
        self._explanation_module = _OnnxExplanationModule(self)

    @property
    def hardware_label(self) -> str:
        primary_provider = self.providers[0] if self.providers else "CPUExecutionProvider"
        return _onnx_hardware_label(primary_provider)

    @classmethod
    def from_path(cls, path: Path, *, hardware: str = "gpu") -> OnnxBackend:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError(_ONNX_RUNTIME_INSTALL_HINT) from error

        providers = resolve_onnx_providers(
            hardware,
            available_providers=ort.get_available_providers(),
        )
        session = ort.InferenceSession(str(path), providers=providers)
        return cls(session, providers=providers, model_path=path)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(
                f"OnnxBackend expected torch.Tensor inputs, got {type(inputs).__name__}."
            )

        input_array = self._tensor_to_numpy(inputs)
        output_names: list[str] | None = self.output_names or None
        outputs = self.session.run(output_names, {self.input_name: input_array})
        tensor_outputs = [torch.from_numpy(np.asarray(output)) for output in outputs]
        if not tensor_outputs:
            raise ValueError("ONNX Runtime returned no outputs.")
        if len(tensor_outputs) == 1:
            return tensor_outputs[0]
        return self._select_primary_output(tensor_outputs)

    def as_model_for_explanation(self) -> nn.Module:
        return self._explanation_module

    def _tensor_to_numpy(self, inputs: torch.Tensor) -> np.ndarray[Any, Any]:
        array = inputs.detach().cpu().numpy()
        expected_dtype = _NUMPY_DTYPES_BY_ONNX_TYPE.get(self.input_type)
        if expected_dtype is None:
            return array
        if array.dtype != expected_dtype:
            return array.astype(expected_dtype, copy=False)
        return array

    @staticmethod
    def _select_primary_output(outputs: list[torch.Tensor]) -> torch.Tensor:
        for tensor in outputs:
            if tensor.ndim >= 2:
                return tensor
        return max(outputs, key=lambda tensor: tensor.numel())


def _move_tensors_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_tensors_to_device(item, device) for key, item in value.items()}
    return value


def _torch_hardware_label(device: torch.device) -> str:
    label_by_type = {
        "cpu": "CPU",
        "cuda": "CUDA",
        "xpu": "Intel XPU",
        "mps": "Apple MPS",
    }
    return label_by_type.get(device.type, device.type.upper())


def _onnx_hardware_label(provider: str) -> str:
    label_by_provider = {
        "CPUExecutionProvider": "CPU",
        "CUDAExecutionProvider": "CUDA",
        "OpenVINOExecutionProvider": "Intel OpenVINO",
        "CoreMLExecutionProvider": "Apple CoreML",
    }
    return label_by_provider.get(provider, provider)
