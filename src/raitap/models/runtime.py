from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_VALID_HARDWARE = frozenset({"cpu", "gpu"})


def validate_hardware(hardware: str) -> str:
    if hardware not in _VALID_HARDWARE:
        valid_values = ", ".join(sorted(_VALID_HARDWARE))
        raise ValueError(f"Invalid hardware {hardware!r}. Expected one of: {valid_values}.")
    return hardware


def resolve_torch_device(hardware: str) -> torch.device:
    resolved_hardware = validate_hardware(hardware)
    if resolved_hardware == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if _torch_xpu_is_available():
        return torch.device("xpu")

    logger.warning(
        "GPU was requested for PyTorch, but neither CUDA nor Intel XPU is available. "
        "Falling back to CPU."
    )
    return torch.device("cpu")


def resolve_onnx_providers(
    hardware: str,
    *,
    available_providers: Sequence[str] | None = None,
) -> list[str]:
    resolved_hardware = validate_hardware(hardware)
    if resolved_hardware == "cpu":
        return ["CPUExecutionProvider"]

    provider_names = available_providers
    if provider_names is None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError(
                "ONNX support is enabled but onnxruntime is not installed. "
                "Install it with `uv sync --extra onnx-cpu`, `uv sync --extra onnx-gpu`, "
                "or `uv sync --extra onnx-openvino`."
            ) from error
        provider_names = ort.get_available_providers()

    if "CUDAExecutionProvider" in provider_names:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if "OpenVINOExecutionProvider" in provider_names:
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    logger.warning(
        "GPU was requested for ONNX Runtime, but neither CUDAExecutionProvider nor "
        "OpenVINOExecutionProvider is available. Falling back to CPUExecutionProvider."
    )
    return ["CPUExecutionProvider"]


def _torch_xpu_is_available() -> bool:
    xpu_module = getattr(torch, "xpu", None)
    if xpu_module is None:
        return False

    is_available = getattr(xpu_module, "is_available", None)
    return bool(callable(is_available) and is_available())
