"""PyTorch-backed model runtime + device/detection helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raitap.models.base_backend import ModelBackend, _adapt_input_shape
from raitap.models.registration import register
from raitap.types import Capability, ResolvedHardware, TaskKind
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from torch import nn
else:
    torch = lazy_import("torch")
    nn = lazy_import("torch.nn")


def _is_torchvision_detection_model(model: nn.Module) -> bool:
    """Return True for torchvision detection models (Faster R-CNN, RetinaNet, SSD)."""
    try:
        from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
        from torchvision.models.detection.retinanet import RetinaNet
        from torchvision.models.detection.ssd import SSD
    except ImportError:
        return False
    return isinstance(model, GeneralizedRCNN | RetinaNet | SSD)


@register(
    provides={Capability.AUTOGRAD},
    extensions={".pth", ".pt"},
    extra="torch",
    supported_hardware={ResolvedHardware.cpu, ResolvedHardware.cuda, ResolvedHardware.xpu},
)
class TorchBackend(ModelBackend):
    """PyTorch-backed model runtime."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | None = None,
        task_kind: TaskKind | None = None,
        category_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device("cpu") if device is None else device
        self._task_kind = task_kind if task_kind is not None else self._infer_task_kind(model)
        self.category_names = category_names
        # ``load_hf_text_backend`` sets this to an ``AutoTokenizer`` instance;
        # declared here so pyright sees the attribute on every ``TorchBackend``
        # (harmless ``None`` otherwise).
        self.tokenizer: Any = None

    @classmethod
    def from_path(
        cls, path: Path, *, model_cfg: Any, hardware: str, allow_unsafe_pickle: bool = False
    ) -> ModelBackend:
        from raitap.models.model import _load_torch_module_from_path  # deferred: import cycle
        from raitap.models.runtime import resolve_torch_device

        device = resolve_torch_device(hardware)
        module = _load_torch_module_from_path(
            path, model_cfg=model_cfg, device=device, allow_unsafe_pickle=allow_unsafe_pickle
        )
        return cls(module, device=device, task_kind=getattr(model_cfg, "task_kind", None))

    @staticmethod
    def _infer_task_kind(model: nn.Module) -> TaskKind:
        from raitap.task_families import TASK_FAMILIES

        matches = [
            family.kind
            for family in TASK_FAMILIES.values()
            if getattr(family, "matches_model", None) and family.matches_model(model)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Model matches multiple task families {sorted(k.value for k in matches)}; "
                "set model.task_kind explicitly to disambiguate."
            )
        return matches[0] if matches else TaskKind.classification

    @property
    def task_kind(self) -> TaskKind:
        return self._task_kind

    @property
    def hardware_label(self) -> str:
        return _torch_hardware_label(self.device)

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        reshaped = _adapt_input_shape(inputs, self.expected_input_shape)
        return reshaped.to(self.device)

    def prepare_detection_inputs(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Prepare a ragged list of detection images for inference.

        Each ``(C, H, W)`` tensor is moved to the backend device without
        any shape adaptation -- detection images keep their native resolution.
        Torchvision detection models accept ``list[Tensor]`` natively.
        """
        return [img.to(self.device) for img in inputs]

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return _move_tensors_to_device(kwargs, self.device)

    def __call__(self, inputs: torch.Tensor | list[torch.Tensor], **kwargs: Any) -> Any:
        # ``**kwargs`` covers text models' ``attention_mask`` (and similar HF
        # keyword-only forward args); image/tabular callers never pass any.
        if kwargs:
            return self.model(inputs, **self._prepare_kwargs(kwargs))
        return self.model(inputs)

    def autograd_module(self) -> nn.Module:
        return self.model


def _move_tensors_to_device(value: Any, device: torch.device | None) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_tensors_to_device(item, device) for key, item in value.items()}
    return value


def _torch_hardware_label(device: torch.device | None) -> str:
    if device is None:
        return "CPU"
    label_by_type = {
        "cpu": "CPU",
        "cuda": "CUDA",
        "xpu": "Intel XPU",
        "mps": "Apple MPS",
    }
    return label_by_type.get(device.type, device.type.upper())


def load_hf_text_backend(source: str, *, tokenizer: str, device: torch.device) -> TorchBackend:
    """Load a HuggingFace sequence-classification model + tokenizer.

    ``source`` and ``tokenizer`` are HuggingFace hub ids or local directories,
    resolved via ``AutoModelForSequenceClassification``/``AutoTokenizer``.
    Returns a ``TorchBackend`` with ``.tokenizer`` set.
    """
    # lazy: `transformers` is an optional dep (extra "text"), imported only
    # when a text model is actually requested.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    module = AutoModelForSequenceClassification.from_pretrained(source).to(device).eval()
    backend = TorchBackend(module, device=device, task_kind=TaskKind.classification)
    backend.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    return backend
