from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torchvision import models

from raitap.tracking.base_tracker import BaseTracker, Trackable

from .backend import ModelBackend, OnnxBackend, TorchBackend
from .runtime import resolve_torch_device

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig


class Model(Trackable):
    def __init__(self, config: AppConfig) -> None:
        self.backend = self._load_model(config)

    def _load_model(self, config: AppConfig) -> ModelBackend:
        source = config.model.source
        hardware = getattr(config, "hardware", "gpu")
        if not source:
            raise ValueError(
                "No model specified. Set model.source in your config.\n"
                "  model.source: path/to/your_model.pth   (custom model)\n"
                "  model.source: resnet50                 (built-in demo model)"
            )

        path = Path(source)
        suffix = path.suffix.lower()

        if path.exists() or suffix:
            return _load_from_path(path, model_cfg=config.model, hardware=hardware)

        name = str(source).lower()
        if hasattr(models, name):
            return _load_pretrained(name, hardware=hardware)

        raise ValueError(
            f"Model source {source!r} is neither an existing path nor a known "
            f"torchvision model.\n"
            f"Supported file formats: {_supported_model_formats()}\n"
            f"To use your own model, set source to a valid model file path."
        )

    def log(self, tracker: BaseTracker, **kwargs: Any) -> None:
        tracker.log_model(self.backend)


def _load_from_path(path: Path, *, model_cfg: Any, hardware: str) -> ModelBackend:
    """
    Load a model backend from a file path.

    Args:
        path: Path to the model file.
        model_cfg: ``ModelConfig``-shaped object providing ``arch``,
            ``num_classes``, and ``pretrained`` for state-dict loading.
        hardware: Hardware label resolved into a device.

    Returns:
        Model backend ready for inference and explanation.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is unsupported.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix.lower() == ".onnx":
        return OnnxBackend.from_path(path, hardware=hardware)

    if path.suffix.lower() in {".pth", ".pt"}:
        device = resolve_torch_device(hardware)
        module = _load_torch_module_from_path(path, model_cfg=model_cfg, device=device)
        return TorchBackend(module, device=device)

    raise ValueError(
        f"Unsupported model format {path.suffix!r}. Supported formats: {_supported_model_formats()}"
    )


def _try_torchscript_load(path: Path) -> nn.Module | None:
    """Try to load *path* as a TorchScript archive; return ``None`` if it isn't one."""
    try:
        scripted = torch.jit.load(str(path), map_location="cpu")
    except RuntimeError:
        # Not a TorchScript archive — fall back to the regular torch.load path.
        return None
    return scripted


def _build_arch_from_config(model_cfg: Any) -> nn.Module:
    """Instantiate a torchvision architecture from ``model_cfg`` for state-dict loading."""
    arch = getattr(model_cfg, "arch", None)
    num_classes = getattr(model_cfg, "num_classes", None)
    pretrained = bool(getattr(model_cfg, "pretrained", False))

    missing = [
        name for name, value in (("arch", arch), ("num_classes", num_classes)) if value is None
    ]
    if missing:
        raise ValueError(
            "State-dict loading requires model."
            + " and model.".join(missing)
            + ". Set them in your config, e.g.:\n"
            "  model:\n"
            "    source: path/to/weights.pth\n"
            "    arch: resnet18\n"
            "    num_classes: 2"
        )

    factory = getattr(models, arch, None)
    if factory is None:
        raise ValueError(f"model.arch {arch!r} is not a known torchvision model.")

    weights = "DEFAULT" if pretrained else None
    return factory(weights=weights, num_classes=num_classes)


def _load_torch_module_from_path(path: Path, *, model_cfg: Any, device: torch.device) -> nn.Module:
    scripted = _try_torchscript_load(path)
    if scripted is not None:
        scripted.to(device)
        scripted.eval()
        return scripted

    # Try the safe path first: `weights_only=True` only deserialises tensors and
    # state-dicts, refusing arbitrary pickled objects (no code execution risk).
    # Pickled `nn.Module` checkpoints fail this and fall through to the
    # deprecated unsafe path below.
    pickled_module = False
    try:
        obj: Any = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        pickled_module = True
        obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        module = _build_arch_from_config(model_cfg)
        module.load_state_dict(obj, strict=True)
        module.to(device)
        module.eval()
        return module

    if isinstance(obj, nn.Module):
        if pickled_module:
            warnings.warn(
                f"Loading pickled nn.Module from {path}: this format is fragile across "
                "environments and torchvision versions, and requires unsafe pickle "
                "deserialisation. Prefer `torch.save(model.state_dict(), path)` with "
                "model.arch + model.num_classes set in the config, or "
                "`torch.jit.save(scripted, path)`.",
                DeprecationWarning,
                stacklevel=2,
            )
        obj.to(device)
        obj.eval()
        return obj

    raise ValueError(f"Expected an nn.Module or state-dict in {path}, got {type(obj).__name__}.")


def _load_pretrained(model_name: str, *, hardware: str) -> ModelBackend:
    """
    Load a torchvision model with its default pre-trained weights.

    This is intended for demos and quick testing.  For production use, supply
    a model file path via :class:`Model` instead.

    Args:
        model_name: Any ``torchvision.models`` attribute name
                    (e.g. ``"resnet50"``, ``"vit_b_32"``).

    Returns:
        PyTorch model in evaluation mode.

    Raises:
        ValueError: If *model_name* is not a valid torchvision model.
    """
    factory = getattr(models, model_name, None)
    if factory is None:
        raise ValueError(f"'{model_name}' is not a known torchvision model.")

    device = resolve_torch_device(hardware)
    model = factory(weights="DEFAULT")
    model.to(device)
    model.eval()
    return TorchBackend(model, device=device)


def _supported_model_formats() -> list[str]:
    return [".onnx", ".pt", ".pth"]
