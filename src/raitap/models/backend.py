from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

    import onnxruntime as ort


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

    @abstractmethod
    def __call__(self, inputs: torch.Tensor) -> Any:
        """Run inference for ``inputs``."""

    @abstractmethod
    def as_model_for_explanation(self) -> nn.Module:
        """Return the model object that explainers should consume."""


class TorchBackend(ModelBackend):
    """PyTorch-backed model runtime."""

    supports_torch_autograd = True

    def __init__(self, model: nn.Module) -> None:
        self.model = model

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

    def __init__(self, session: ort.InferenceSession, *, model_path: Path | None = None) -> None:
        self.session = session
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

    @classmethod
    def from_path(cls, path: Path) -> OnnxBackend:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError(
                "ONNX support is enabled but onnxruntime is not installed. "
                "Install it with `uv sync --extra onnx`."
            ) from error

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return cls(session, model_path=path)

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
