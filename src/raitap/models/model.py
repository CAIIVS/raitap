from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

from raitap import raitap_log
from raitap.configs import cfg_to_dict
from raitap.data.metadata import shape_tuple
from raitap.data.preprocessing import ResolvedPreprocessing, resolve_preprocessing
from raitap.tracking.base_tracker import BaseTracker, Trackable
from raitap.utils.lazy import lazy_import

from .backend import ModelBackend, OnnxBackend, TorchBackend
from .runtime import resolve_torch_device

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torchvision import models

    from raitap.configs.schema import AppConfig
else:
    torch = lazy_import("torch")
    nn = lazy_import("torch.nn")
    models = lazy_import("torchvision.models")


class Model(Trackable):
    def __init__(
        self,
        config: AppConfig,
        *,
        resolved_preprocessing: ResolvedPreprocessing | None = None,
        allow_unsafe_pickle: bool = False,
    ) -> None:
        self.backend = self._load_model(config, allow_unsafe_pickle=allow_unsafe_pickle)
        self.resolved_preprocessing = _apply_preprocessing(
            self.backend,
            config,
            resolved_preprocessing=resolved_preprocessing,
        )
        shape_override = _resolve_shape_override(config)
        if shape_override is not None:
            self.backend.expected_input_shape = shape_override

    def _load_model(self, config: AppConfig, *, allow_unsafe_pickle: bool = False) -> ModelBackend:
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
            return _load_from_path(
                path,
                model_cfg=config.model,
                hardware=hardware,
                allow_unsafe_pickle=allow_unsafe_pickle,
            )

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


def _apply_preprocessing(
    backend: ModelBackend,
    config: AppConfig,
    *,
    resolved_preprocessing: ResolvedPreprocessing | None = None,
) -> ResolvedPreprocessing:
    """Resolve ``data.preprocessing`` and wrap the backend's model in-place
    with the value half (Normalize).

    The shape half (Resize / CenterCrop) is applied per-image by
    :func:`raitap.data.data._load_data` before stacking, so by the time we
    see the model, inputs already have a uniform shape.

    ONNX models are not wrapped here — the resolver still reports the chosen
    option for transparency, but a non-None ``model_module`` on an ONNX
    backend raises so the user knows to pre-normalize externally.
    """
    resolved = (
        resolved_preprocessing
        if resolved_preprocessing is not None
        else resolve_preprocessing(config.model, config.data)
    )

    if resolved.model_module is not None:
        if isinstance(backend, OnnxBackend):
            raise NotImplementedError(
                "data.preprocessing is not yet supported for ONNX models. "
                "Pre-normalize your inputs externally, or convert the model "
                "to PyTorch (.pt / .pth / TorchScript)."
            )
        if isinstance(backend, TorchBackend):
            # ``to`` and ``eval`` mutate nn.Module instances; keep the
            # ResolvedPreprocessing record stable for later readers.
            model_module = copy.deepcopy(resolved.model_module).to(backend.device)
            model_module.eval()
            backend.model = nn.Sequential(model_module, backend.model)

    for warning in resolved.warnings:
        raitap_log.warn(warning)

    if resolved.is_active:
        raitap_log.info(f"Preprocessing: {resolved.description}")

    return resolved


def _resolve_shape_override(config: Any) -> tuple[int | None, ...] | None:
    """
    Extract a user-specified non-batch input shape from
    ``config.data.input_metadata.shape`` and convert it into an
    ``expected_input_shape`` tuple with a dynamic batch dimension prepended.

    Returns ``None`` if the config does not specify a shape, allowing callers
    to keep whatever the backend resolved on its own. Parsing is delegated to
    :func:`raitap.data.metadata.shape_tuple` so semantics stay consistent with
    the rest of the metadata pipeline.
    """
    data_cfg = getattr(config, "data", None)
    if data_cfg is None:
        return None
    input_metadata = getattr(data_cfg, "input_metadata", None)
    if input_metadata is None:
        return None
    try:
        metadata = cfg_to_dict(input_metadata)
    except Exception:
        return None
    if not isinstance(metadata, dict):
        return None
    non_batch = shape_tuple(metadata.get("shape"))
    if not non_batch:
        return None
    return (None, *non_batch)


def _load_from_path(
    path: Path,
    *,
    model_cfg: Any,
    hardware: str,
    allow_unsafe_pickle: bool = False,
) -> ModelBackend:
    """
    Load a model backend from a file path.

    Args:
        path: Path to the model file.
        model_cfg: ``ModelConfig``-shaped object providing ``arch``,
            ``num_classes``, and ``pretrained`` for state-dict loading.
        hardware: Hardware label resolved into a device.
        allow_unsafe_pickle: Explicit consent to deserialise a pickled
            ``nn.Module`` checkpoint (executes arbitrary code embedded in
            the file). Mirrors the ``--allow-unsafe-pickle`` CLI flag.

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
        module = _load_torch_module_from_path(
            path,
            model_cfg=model_cfg,
            device=device,
            allow_unsafe_pickle=allow_unsafe_pickle,
        )
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
    arch: str | None = getattr(model_cfg, "arch", None)
    num_classes: int | None = getattr(model_cfg, "num_classes", None)
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
    assert arch is not None and num_classes is not None  # narrowed by `missing` check

    factory = getattr(models, arch, None)
    if factory is None:
        raise ValueError(f"model.arch {arch!r} is not a known torchvision model.")

    weights = "DEFAULT" if pretrained else None
    return factory(weights=weights, num_classes=num_classes)


def _load_torch_module_from_path(
    path: Path,
    *,
    model_cfg: Any,
    device: torch.device,
    allow_unsafe_pickle: bool = False,
) -> nn.Module:
    scripted = _try_torchscript_load(path)
    if scripted is not None:
        scripted.to(device)
        scripted.eval()
        return scripted

    # Try the safe path first: `weights_only=True` only deserialises tensors and
    # state-dicts, refusing arbitrary pickled objects (no code execution risk).
    # Pickled `nn.Module` checkpoints fail this with `pickle.UnpicklingError`
    # ("Weights only load failed ... Unsupported global ...") and require the
    # unsafe path, which executes arbitrary code embedded in the file. We
    # refuse it unless the user has explicitly opted in via the
    # ``allow_unsafe_pickle`` kwarg on :func:`raitap.run` (Python API) or the
    # ``--allow-unsafe-pickle`` CLI flag (surfaced through the
    # ``RAITAP_ALLOW_UNSAFE_PICKLE`` env var the bootstrap exports across
    # re-exec). Any other exception (corrupted archive, I/O error, version
    # incompatibility) is re-raised as-is so the real failure mode is not
    # masked by the unsafe-pickle guidance.
    consent = allow_unsafe_pickle or os.environ.get("RAITAP_ALLOW_UNSAFE_PICKLE") == "1"
    pickled_module = False
    try:
        obj: Any = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as safe_load_error:
        if not consent:
            raise ValueError(
                f"Refusing to load {path}: the file requires unsafe pickle "
                "deserialisation, which executes arbitrary code embedded in "
                "the checkpoint. Re-save it as a state-dict "
                "(`torch.save(model.state_dict(), path)` plus model.arch + "
                "model.num_classes in the config) or a TorchScript archive "
                "(`torch.jit.save(scripted, path)`). If the source is fully "
                "trusted, pass `allow_unsafe_pickle=True` to `raitap.run(...)` "
                "or re-run the CLI with `--allow-unsafe-pickle`. "
                f"Underlying error: {safe_load_error}"
            ) from safe_load_error
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
            raitap_log.warn(
                f"Loading pickled nn.Module from {path}: this format is fragile across "
                "environments and torchvision versions, and requires unsafe pickle "
                "deserialisation. Prefer `torch.save(model.state_dict(), path)` with "
                "model.arch + model.num_classes set in the config, or "
                "`torch.jit.save(scripted, path)`.",
                category=DeprecationWarning,
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
