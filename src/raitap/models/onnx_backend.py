"""ONNX Runtime-backed model runtime + onnx shape/dtype helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from raitap.models.base_backend import ModelBackend, _adapt_input_shape
from raitap.models.registration import register
from raitap.types import FORWARD_ONLY, ResolvedHardware
from raitap.utils.errors import ModelInputShapeError
from raitap.utils.lazy import lazy_import

from .runtime import _ONNX_RUNTIME_INSTALL_HINT, resolve_onnx_providers

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import onnxruntime as ort
    import torch
    from torch import nn
else:
    torch = lazy_import("torch")
    nn = lazy_import("torch.nn")


def _resolve_onnx_expected_shape(
    raw_shape: Sequence[Any],
) -> tuple[int | None, ...] | None:
    """Translate an ONNX input shape spec into a ``(int | None, ...)`` tuple
    suitable for :func:`_adapt_input_shape`.

    ONNX dims can be ints, ``None``, or strings (symbolic names like
    ``"batch"``). Concrete positive ints are respected; ``None`` and
    strings become ``None`` (dynamic, resolved at runtime from the input
    batch dim).

    Returns ``None`` when the shape is empty (scalar input -- no adaptation
    possible). Raises :class:`ModelInputShapeError` when two or more dims
    are dynamic, since reshape targets become ambiguous and the user must
    override via ``data.input_metadata.shape``.
    """
    if not raw_shape:
        return None
    parsed: list[int | None] = []
    for dim in raw_shape:
        if isinstance(dim, int) and dim > 0:
            parsed.append(int(dim))
        else:
            parsed.append(None)
    dynamic_total = sum(1 for dim in parsed if dim is None)
    if dynamic_total >= 2:
        raise ModelInputShapeError(expected_shape=tuple(parsed))
    return tuple(parsed)


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


@register(
    provides=FORWARD_ONLY,
    extensions={".onnx"},
    extra="onnx",
    supported_hardware={ResolvedHardware.cpu, ResolvedHardware.cuda, ResolvedHardware.xpu},
)
class OnnxBackend(ModelBackend):
    """ONNX Runtime-backed model runtime."""

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
        self._preprocessing_module: nn.Module | None = None
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"Expected exactly one ONNX input, got {len(inputs)}. "
                "Multi-input ONNX models are not supported yet."
            )
        self.input_name = inputs[0].name
        self.input_type = inputs[0].type
        self.output_names = [output.name for output in session.get_outputs()]
        self.expected_input_shape = _resolve_onnx_expected_shape(inputs[0].shape)

    @property
    def hardware_label(self) -> str:
        primary_provider = self.providers[0] if self.providers else "CPUExecutionProvider"
        return _onnx_hardware_label(primary_provider)

    def set_preprocessing(self, module: nn.Module | None) -> None:
        """Attach a preprocessing module for tensor calls."""
        if module is not None:
            module.cpu()
            module.eval()
        self._preprocessing_module = module

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Leave shape adaptation to attached preprocessing when present."""
        if self._preprocessing_module is not None:
            return inputs
        return super()._prepare_inputs(inputs)

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        model_cfg: Any = None,
        hardware: str = "gpu",
        allow_unsafe_pickle: bool = False,
    ) -> OnnxBackend:
        # model_cfg / allow_unsafe_pickle unused: ONNX needs neither.
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

    def forward_numpy(self, batch: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Run inference on a NumPy batch (no Torch required on the hot path)."""
        expected_dtype = _NUMPY_DTYPES_BY_ONNX_TYPE.get(self.input_type)
        input_array = _adapt_input_shape(batch, self.expected_input_shape)
        if expected_dtype is not None and input_array.dtype != expected_dtype:
            input_array = input_array.astype(expected_dtype, copy=False)
        output_names: list[str] | None = self.output_names or None
        outputs = self.session.run(output_names, {self.input_name: input_array})
        np_outputs = [np.asarray(output) for output in outputs]
        if not np_outputs:
            raise ValueError("ONNX Runtime returned no outputs.")
        if len(np_outputs) == 1:
            return np_outputs[0]
        return self._select_primary_numpy_output(np_outputs)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(
                f"OnnxBackend expected torch.Tensor inputs, got {type(inputs).__name__}."
            )

        prepared_inputs = inputs
        if self._preprocessing_module is not None:
            with torch.no_grad():
                prepared_inputs = self._preprocessing_module(inputs.detach().cpu())
            if not isinstance(prepared_inputs, torch.Tensor):
                raise TypeError(
                    "OnnxBackend preprocessing expected a torch.Tensor output, "
                    f"got {type(prepared_inputs).__name__}."
                )

        input_array = self._tensor_to_numpy(prepared_inputs)
        primary = self.forward_numpy(input_array)
        return torch.from_numpy(np.asarray(primary))

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

    @staticmethod
    def _select_primary_numpy_output(outputs: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        for arr in outputs:
            if arr.ndim >= 2:
                return arr
        return max(outputs, key=lambda arr: arr.size)


def _onnx_hardware_label(provider: str) -> str:
    label_by_provider = {
        "CPUExecutionProvider": "CPU",
        "CUDAExecutionProvider": "CUDA",
        "OpenVINOExecutionProvider": "Intel OpenVINO",
        "CoreMLExecutionProvider": "Apple CoreML",
    }
    return label_by_provider.get(provider, provider)
