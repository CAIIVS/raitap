"""Resolve the ``data.preprocessing`` config option into ready-to-wrap modules.

Three options (see :class:`raitap.configs.schema.DataConfig.preprocessing`):

- ``None``            → no preprocessing; resolver returns both halves ``None``
                        and a loud warning so the pipeline can surface it.
- ``"model-bundled"`` → resolve via ``torchvision.models.get_model_weights(arch)``
                           and split ``Weights.DEFAULT.transforms()`` into a shape
                           half (Resize + CenterCrop, run per-image in the data
                           loader) and a value half (Normalize, wrapped at the
                           model boundary).
- path to a ``.py`` file → import it via ``importlib.util.spec_from_file_location``
                           and call its ``make_preprocessing()`` factory (model
                           side). Optionally also calls ``make_data_preprocessing()``
                           when the file exports it (data-loader side). Gated by
                           ``data.acknowledge_preprocessing_exec`` (Python API) or
                           ``--allow-preprocessing-exec``/``-yp`` (CLI, surfaced
                           through the ``RAITAP_ALLOW_PREPROCESSING_EXEC`` env var).

Split rationale: shape preprocessing (Resize, CenterCrop) must run per-image in
the data loader so mixed-size directories can be stacked at all; it has no
gradient so leaving it outside the autograd graph is safe. Value preprocessing
(Normalize) must stay inside autograd so attribution and adversarial budgets
operate on the same ``[0, 1]`` input space the user sees.

This module is side-effect-free: it returns a :class:`ResolvedPreprocessing`
record. Callers (currently :class:`raitap.models.model.Model` for the model
half and :func:`raitap.data.data._load_data` for the data half) are responsible
for emitting the user-facing panel / warning via ``raitap_log``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from torch import nn
    from torchvision import models
    from torchvision.transforms import _presets, v2

    from raitap.configs.schema import DataConfig, ModelConfig
else:
    torch = lazy_import("torch")
    nn = lazy_import("torch.nn")
    models = lazy_import("torchvision.models")
    v2 = lazy_import("torchvision.transforms.v2")
    _presets = lazy_import("torchvision.transforms._presets")

_CONSENT_ENV_VAR = "RAITAP_ALLOW_PREPROCESSING_EXEC"
_FACTORY_NAME = "make_preprocessing"
_DATA_FACTORY_NAME = "make_data_preprocessing"
_MODEL_BUNDLED_VALUE = "model-bundled"

Origin = Literal["off", "model-bundled", "custom-file"]


@dataclass(frozen=True)
class ResolvedPreprocessing:
    """Single source of truth for everything the pipeline needs to know about
    the resolved preprocessing setup. Consumed by the data loader (per-image
    shape transform), the model wrap (per-batch value transform), the startup
    panel, the HTML report card, and MLflow params.
    """

    data_module: nn.Module | None
    model_module: nn.Module | None
    origin: Origin
    description: str
    file_path: Path | None = None
    file_sha256: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.data_module is not None or self.model_module is not None


def resolve_preprocessing(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
) -> ResolvedPreprocessing:
    """Resolve ``data_cfg.preprocessing`` into a :class:`ResolvedPreprocessing`.

    No side effects. Callers emit the panel / warnings.
    """
    raw = getattr(data_cfg, "preprocessing", None)

    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _resolve_off(data_cfg)

    if not isinstance(raw, str):
        raise TypeError(
            f"data.preprocessing must be a string or null, got {type(raw).__name__}. "
            f"Use 'model-bundled' or a path to a .py file."
        )

    if raw == _MODEL_BUNDLED_VALUE:
        return _resolve_model_bundled(model_cfg)

    return _resolve_custom_file(raw, data_cfg)


def module_as_per_image_callable(
    module: nn.Module | None,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Lift an ``nn.Module`` into a per-image callable for loader-side transforms.

    The data half of preprocessing runs before images are stacked, so callers
    need a single-image ``(C, H, W) -> Tensor`` callable. The module is put in
    eval mode and run without autograd because shape preprocessing is outside
    the attribution/attack graph by design.
    """
    if module is None:
        return None
    module.eval()

    def _apply(image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return module(image)

    return _apply


# ---------------------------------------------------------------------------
# Off
# ---------------------------------------------------------------------------


def _resolve_off(data_cfg: DataConfig) -> ResolvedPreprocessing:
    warnings: list[str] = []
    if not _suppress_off_warning(data_cfg):
        warnings.append(_OFF_WARNING)
    return ResolvedPreprocessing(
        data_module=None,
        model_module=None,
        origin="off",
        description="No preprocessing applied",
        warnings=warnings,
    )


def _suppress_off_warning(data_cfg: DataConfig) -> bool:
    if bool(getattr(data_cfg, "acknowledge_preprocessing_off", False)):
        return True
    input_metadata = getattr(data_cfg, "input_metadata", None)
    if input_metadata is None:
        return False
    # ``input_metadata`` may be a plain dict (Python API) or an OmegaConf
    # DictConfig (Hydra/YAML). Route through ``cfg_to_dict`` so both produce
    # the same suppression decision — otherwise tabular CLI runs leak the
    # image-only warning.
    try:
        from raitap.configs.utils import cfg_to_dict

        metadata = cfg_to_dict(input_metadata)
    except Exception:
        return False
    if not isinstance(metadata, dict):
        return False
    kind = metadata.get("kind")
    return kind is not None and str(kind) != "image"


_OFF_WARNING = (
    "Image preprocessing is OFF.\n"
    "Raitap is forwarding your images to the model unchanged.\n"
    "Most pretrained image models expect normalized inputs — without\n"
    "preprocessing, your accuracy results may be silently incorrect.\n"
    "\n"
    "Choose a preprocessing option in your config:\n"
    "    data:\n"
    "      preprocessing: model-bundled          # use the model's bundled preprocessing\n"
    "    # OR\n"
    "    data:\n"
    "      preprocessing: ./preprocessing.py        # supply your own\n"
    "\n"
    "If you've already preprocessed your images, silence this message:\n"
    "    data:\n"
    "      acknowledge_preprocessing_off: true"
)


# ---------------------------------------------------------------------------
# Model-bundled
# ---------------------------------------------------------------------------


def _resolve_model_bundled(model_cfg: ModelConfig) -> ResolvedPreprocessing:
    arch = _arch_from_model_cfg(model_cfg)
    if arch is None:
        raise ValueError(
            "data.preprocessing: 'model-bundled' requires a torchvision "
            "arch. Set model.arch (e.g. resnet50) or model.source to a built-in "
            "torchvision name. For ONNX or pickled models, use "
            "data.preprocessing: ./your_preprocessing.py instead."
        )

    try:
        weights_enum = models.get_model_weights(arch)
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"data.preprocessing: 'model-bundled' could not resolve weights "
            f"for arch {arch!r}: {exc}. Use a path to a .py file instead."
        ) from exc

    default = weights_enum["DEFAULT"]
    preset = default.transforms()
    data_module, model_module = _split_preset(preset)
    description = f"{weights_enum.__name__}.DEFAULT ({default.name})"

    return ResolvedPreprocessing(
        data_module=data_module,
        model_module=model_module,
        origin="model-bundled",
        description=description,
    )


def _arch_from_model_cfg(model_cfg: ModelConfig) -> str | None:
    arch = getattr(model_cfg, "arch", None)
    if isinstance(arch, str) and arch.strip():
        return arch
    source = getattr(model_cfg, "source", None)
    if isinstance(source, str):
        candidate = source.strip().lower()
        if candidate and hasattr(models, candidate):
            attr = getattr(models, candidate)
            if callable(attr):
                return candidate
    return None


def _split_preset(preset: Any) -> tuple[nn.Module | None, nn.Module | None]:
    """Split a torchvision preset into (shape-half, value-half).

    - ``ImageClassification`` → Resize + CenterCrop for the loader, Normalize
      for the model boundary.
    - ``SemanticSegmentation`` → Resize for the loader, Normalize for the
      boundary.
    - ``ObjectDetection`` (and any unrecognised preset that runs as a no-op)
      → both halves ``None``. Detection models normalise internally.
    - Anything else → fall back to model-side only.
    """
    if isinstance(preset, _presets.ImageClassification):
        data_half = nn.Sequential(
            v2.Resize(
                list(preset.resize_size),
                interpolation=preset.interpolation,
                antialias=preset.antialias,
            ),
            v2.CenterCrop(list(preset.crop_size)),
        )
        model_half = v2.Normalize(mean=list(preset.mean), std=list(preset.std))
        return data_half, model_half

    if isinstance(preset, _presets.SemanticSegmentation):
        # Torchvision's segmentation preset stores ``resize_size`` as a
        # sequence (e.g. ``[520]``) which ``v2.Resize`` interprets as
        # shortest-edge resize that preserves aspect ratio. Pass it through
        # as-is so behaviour matches the preset's ``forward``. Mixed-AR
        # inputs will fail to stack downstream — the loader's error message
        # points to that case.
        resize_size = getattr(preset, "resize_size", None)
        if resize_size is None:
            data_half = None
        else:
            data_half = v2.Resize(
                list(resize_size),
                interpolation=preset.interpolation,
                antialias=preset.antialias,
            )
        model_half = v2.Normalize(mean=list(preset.mean), std=list(preset.std))
        return data_half, model_half

    # ObjectDetection and unknown presets: forward unchanged. Detection models
    # do their own normalisation inside ``GeneralizedRCNNTransform``.
    if _is_noop_preset(preset):
        return None, None

    # Fallback for unrecognised presets — keep them at the model boundary so
    # behaviour at least matches the bundled preset, even if mixed-size dirs
    # would fail upstream.
    return None, _preset_wrapper_cls()(preset)


def _is_noop_preset(preset: Any) -> bool:
    return type(preset).__name__ == "ObjectDetection"


_PRESET_WRAPPER_CLS: type | None = None


def _preset_wrapper_cls() -> type:
    # Lazy class factory — ``nn.Module`` as a base is evaluated at class-def
    # time, which would force a real ``import torch`` at module load and break
    # the partial-extras-venv contract (see ``raitap.utils.lazy``). The wrapper
    # lifts a torchvision transforms preset to an ``nn.Module`` so it composes
    # with ``nn.Sequential(value, backbone)``, transparent to autograd /
    # attribution / attacks.
    global _PRESET_WRAPPER_CLS
    if _PRESET_WRAPPER_CLS is not None:
        return _PRESET_WRAPPER_CLS

    class _PresetWrapper(nn.Module):
        def __init__(self, preset: Any) -> None:
            super().__init__()
            self.preset = preset

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.preset(x)

    _PRESET_WRAPPER_CLS = _PresetWrapper
    return _PresetWrapper


# ---------------------------------------------------------------------------
# Custom file
# ---------------------------------------------------------------------------


def _resolve_custom_file(raw_path: str, data_cfg: DataConfig) -> ResolvedPreprocessing:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"data.preprocessing: file not found at {path}. "
            f"Provide a path to a .py file that exports `{_FACTORY_NAME}()`."
        )
    if path.suffix.lower() != ".py":
        raise ValueError(
            f"data.preprocessing: expected a .py file, got {path.suffix!r}. "
            f"Use 'model-bundled' or a Python file path."
        )

    if not _consent_given(data_cfg):
        raise PermissionError(
            f"Refusing to run {raw_path} because it is arbitrary Python code.\n"
            "\n"
            "To allow it, choose one:\n"
            "  - CLI:          re-run with --allow-preprocessing-exec (short: -yp)\n"
            "  - Python API:   set data.acknowledge_preprocessing_exec: true "
            "on your config\n"
            "\n"
            "If you only need standard ImageNet preprocessing, use the model's\n"
            "bundled preprocessing instead:\n"
            "    data:\n"
            "      preprocessing: model-bundled"
        )

    file_sha256 = _sha256_of_file(path)
    user_module = _import_user_module(path)

    model_module = _build_user_factory(user_module, path, _FACTORY_NAME, required=True)
    data_module = _build_user_factory(user_module, path, _DATA_FACTORY_NAME, required=False)
    if data_module is not None and data_module is model_module:
        raise ValueError(
            f"{path}: `{_FACTORY_NAME}()` and `{_DATA_FACTORY_NAME}()` must return "
            "separate nn.Module instances. The model preprocessing module may be "
            "moved to the model device, while the data preprocessing module runs "
            "on CPU image tensors."
        )

    return ResolvedPreprocessing(
        data_module=data_module,
        model_module=model_module,
        origin="custom-file",
        description=f"Custom file: {raw_path}",
        file_path=path,
        file_sha256=file_sha256,
    )


def _consent_given(data_cfg: DataConfig) -> bool:
    if bool(getattr(data_cfg, "acknowledge_preprocessing_exec", False)):
        return True
    return os.environ.get(_CONSENT_ENV_VAR) == "1"


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_user_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("raitap_user_preprocessing", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_user_factory(
    user_module: Any,
    path: Path,
    factory_name: str,
    *,
    required: bool,
) -> nn.Module | None:
    factory = getattr(user_module, factory_name, None)
    if factory is None or not callable(factory):
        if required:
            raise AttributeError(
                f"{path}: missing callable `{factory_name}()`. "
                f"Export `def {factory_name}() -> nn.Module:` from the file."
            )
        return None
    result = factory()
    if not isinstance(result, nn.Module):
        raise TypeError(
            f"{path}: `{factory_name}()` must return an nn.Module, got {type(result).__name__}."
        )
    return result
