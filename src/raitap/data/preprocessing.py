"""Resolve the two preprocessing knobs into ready-to-wrap modules.

Two independent options on :class:`raitap.configs.schema.DataConfig`:

- ``data.preprocessing`` selects the data-side stage that runs per-image in
  the loader, before the batch is stacked. Typical contents: Resize +
  CenterCrop. No autograd — shape changes don't need gradients, and per-image
  execution is what lets mixed-size folders stack at all.
- ``data.model_input_transformation`` selects the stage applied at the model
  boundary on every forward pass. Typical contents: Normalize. Stays inside
  autograd so attribution and adversarial budgets see the user-facing input
  space.

Each knob accepts ``None``, ``"model-bundled"``, or a path to a ``.py`` file
with the matching RAITAP decorator
(:func:`raitap_preprocessing_factory` for the data side,
:func:`raitap_model_input_transformation_factory` for the model side).

When both knobs are ``None`` and inputs are images, a loud warning fires so
users don't silently feed unnormalised images to pretrained models. The
warning can be silenced for already-preprocessed datasets via
``acknowledge_off``. Custom-file loading on either knob is gated by
``acknowledge_exec``.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from raitap.data.types import Preprocessing
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
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
_ACKNOWLEDGE_OFF_ENV_VAR = "RAITAP_ACKNOWLEDGE_PREPROCESSING_OFF"
_LEGACY_FACTORY_NAMES = ("make_preprocessing", "make_data_preprocessing")
_DATA_PREPROCESSING_FACTORY_ATTR = "__raitap_preprocessing_factory__"
_MODEL_INPUT_TRANSFORMATION_FACTORY_ATTR = "__raitap_model_input_transformation_factory__"
_MODEL_BUNDLED_VALUE = Preprocessing.model_bundled.value

Origin = Literal["off", "model-bundled", "custom-file"]
Side = Literal["data", "model"]
_FactoryT = TypeVar("_FactoryT", bound=Callable[[], nn.Module])


class DataPreprocessingFactory(Protocol):
    """Static contract for data-side preprocessing factories.

    Data preprocessing runs before batching and is typically used for shape
    transforms such as Resize and CenterCrop.
    """

    def __call__(self) -> nn.Module:
        """Build and return the preprocessing module."""
        raise NotImplementedError


class ModelInputTransformationFactory(Protocol):
    """Static contract for model-side input transformation factories.

    Model input transformations run inside every model call and are typically
    used for value transforms such as input normalization.
    """

    def __call__(self) -> nn.Module:
        """Build and return the model input transformation module."""
        raise NotImplementedError


def raitap_preprocessing_factory(factory: _FactoryT) -> _FactoryT:
    """Mark ``factory`` as the custom-file data preprocessing factory."""
    setattr(factory, _DATA_PREPROCESSING_FACTORY_ATTR, True)
    return factory


def raitap_model_input_transformation_factory(factory: _FactoryT) -> _FactoryT:
    """Mark ``factory`` as the custom-file model input transformation factory."""
    setattr(factory, _MODEL_INPUT_TRANSFORMATION_FACTORY_ATTR, True)
    return factory


@dataclass(frozen=True)
class ResolvedPreprocessing:
    """Single source of truth for everything the pipeline needs to know about
    the resolved preprocessing setup. Consumed by the data loader (per-image
    data preprocessing), the model wrap (per-batch model input transformation),
    the startup panel, the HTML report card, and MLflow params.
    """

    data_module: nn.Module | None
    model_module: nn.Module | None
    data_origin: Origin
    model_origin: Origin
    description: str
    data_file_path: Path | None = None
    data_file_sha256: str | None = None
    model_file_path: Path | None = None
    model_file_sha256: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.data_module is not None or self.model_module is not None


@dataclass(frozen=True)
class _SideResolved:
    module: nn.Module | None
    origin: Origin
    description: str | None
    file_path: Path | None = None
    file_sha256: str | None = None
    warnings: tuple[str, ...] = ()


def resolve_preprocessing(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    *,
    acknowledge_off: bool = False,
    acknowledge_exec: bool = False,
) -> ResolvedPreprocessing:
    """Resolve both preprocessing knobs into a :class:`ResolvedPreprocessing`.

    No side effects. Callers emit the panel / warnings.

    ``acknowledge_off`` silences the "preprocessing is OFF" warning for callers
    who have already normalised inputs; mirrors the
    ``--acknowledge-preprocessing-off`` CLI flag. ``acknowledge_exec`` is the
    explicit consent for any ``.py`` file path passed to either knob; mirrors
    the ``--allow-preprocessing-exec``/``-yp`` CLI flag. Either flag may also
    be delivered through its env var (``RAITAP_ACKNOWLEDGE_PREPROCESSING_OFF``
    / ``RAITAP_ALLOW_PREPROCESSING_EXEC``) — that is the bridge the CLI uses
    across its bootstrap re-exec.
    """
    data_raw = _normalise_raw(getattr(data_cfg, "preprocessing", None), "preprocessing")
    model_raw = _normalise_raw(
        getattr(data_cfg, "model_input_transformation", None), "model_input_transformation"
    )

    if data_raw is None and model_raw is None:
        return _resolve_both_off(data_cfg, acknowledge_off=acknowledge_off)

    file_cache: dict[Path, Any] = {}
    hash_cache: dict[Path, str] = {}
    bundled_cache: dict[str, tuple[nn.Module | None, nn.Module | None, str]] = {}

    data_side = _resolve_side(
        data_raw,
        side="data",
        model_cfg=model_cfg,
        acknowledge_exec=acknowledge_exec,
        file_cache=file_cache,
        hash_cache=hash_cache,
        bundled_cache=bundled_cache,
    )
    model_side = _resolve_side(
        model_raw,
        side="model",
        model_cfg=model_cfg,
        acknowledge_exec=acknowledge_exec,
        file_cache=file_cache,
        hash_cache=hash_cache,
        bundled_cache=bundled_cache,
    )

    _reject_shared_module(data_side, model_side)

    warnings: list[str] = []
    warnings.extend(data_side.warnings)
    warnings.extend(model_side.warnings)
    if (
        model_side.module is None
        and data_side.module is not None
        and not _suppress_off_warning(data_cfg, acknowledge_off=acknowledge_off)
    ):
        warnings.append(_MODEL_OFF_WARNING)

    return ResolvedPreprocessing(
        data_module=data_side.module,
        model_module=model_side.module,
        data_origin=data_side.origin,
        model_origin=model_side.origin,
        description=_compose_description(data_side, model_side),
        data_file_path=data_side.file_path,
        data_file_sha256=data_side.file_sha256,
        model_file_path=model_side.file_path,
        model_file_sha256=model_side.file_sha256,
        warnings=warnings,
    )


def module_as_per_image_callable(
    module: nn.Module | None,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Lift an ``nn.Module`` into a per-image callable for loader-side transforms.

    Data preprocessing runs before images are stacked, so callers need a
    single-image ``(C, H, W) -> Tensor`` callable. The module is put in eval
    mode and run without autograd because shape preprocessing is outside the
    attribution/attack graph by design.
    """
    if module is None:
        return None
    module.eval()

    def _apply(image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return module(image)

    return _apply


# ---------------------------------------------------------------------------
# Per-side dispatch
# ---------------------------------------------------------------------------


def _normalise_raw(raw: Any, knob_name: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError(
            f"data.{knob_name} must be a string or null, got {type(raw).__name__}. "
            f"Use 'model-bundled' or a path to a .py file."
        )
    stripped = raw.strip()
    return stripped or None


def _resolve_side(
    raw: str | None,
    *,
    side: Side,
    model_cfg: ModelConfig,
    acknowledge_exec: bool,
    file_cache: dict[Path, Any],
    hash_cache: dict[Path, str],
    bundled_cache: dict[str, tuple[nn.Module | None, nn.Module | None, str]],
) -> _SideResolved:
    if raw is None:
        return _SideResolved(module=None, origin="off", description=None)

    if raw == _MODEL_BUNDLED_VALUE:
        return _resolve_side_bundled(side=side, model_cfg=model_cfg, bundled_cache=bundled_cache)

    return _resolve_side_custom_file(
        raw,
        side=side,
        acknowledge_exec=acknowledge_exec,
        file_cache=file_cache,
        hash_cache=hash_cache,
    )


# ---------------------------------------------------------------------------
# Both off
# ---------------------------------------------------------------------------


def _resolve_both_off(
    data_cfg: DataConfig, *, acknowledge_off: bool = False
) -> ResolvedPreprocessing:
    warnings: list[str] = []
    if not _suppress_off_warning(data_cfg, acknowledge_off=acknowledge_off):
        warnings.append(_OFF_WARNING)
    return ResolvedPreprocessing(
        data_module=None,
        model_module=None,
        data_origin="off",
        model_origin="off",
        description="No preprocessing applied",
        warnings=warnings,
    )


def _suppress_off_warning(data_cfg: DataConfig, *, acknowledge_off: bool = False) -> bool:
    if acknowledge_off or os.environ.get(_ACKNOWLEDGE_OFF_ENV_VAR) == "1":
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
    "RAITAP is forwarding your images to the model unchanged.\n"
    "Most pretrained image models expect normalized inputs — without\n"
    "preprocessing, your accuracy results may be silently incorrect.\n"
    "\n"
    "Set both knobs in your config:\n"
    "    data:\n"
    "      preprocessing: model-bundled                    # data-side (Resize + CenterCrop)\n"
    "      model_input_transformation: model-bundled       # model-side (Normalize)\n"
    "    # OR mix in your own file(s):\n"
    "    data:\n"
    "      preprocessing: ./resize.py\n"
    "      model_input_transformation: ./normalize.py\n"
    "\n"
    "If you've already preprocessed your images, silence this message:\n"
    "  - Python API:   pass acknowledge_preprocessing_off=True to raitap.run(...)\n"
    "  - CLI:          re-run with --acknowledge-preprocessing-off"
)

_MODEL_OFF_WARNING = (
    "data.model_input_transformation is OFF.\n"
    "Data-side preprocessing will run, but no transformation is applied at\n"
    "the model boundary. Pretrained models that expect normalized inputs\n"
    "(ImageNet mean/std, etc.) will produce silently incorrect predictions.\n"
    "\n"
    "Set the model-side knob:\n"
    "    data:\n"
    "      model_input_transformation: model-bundled    # or ./normalize.py"
)


# ---------------------------------------------------------------------------
# Model-bundled
# ---------------------------------------------------------------------------


def _resolve_side_bundled(
    *,
    side: Side,
    model_cfg: ModelConfig,
    bundled_cache: dict[str, tuple[nn.Module | None, nn.Module | None, str]],
) -> _SideResolved:
    arch = _arch_from_model_cfg(model_cfg)
    if arch is None:
        raise ValueError(
            f"data.{_knob_name(side)}: 'model-bundled' requires a torchvision "
            "arch. Set model.arch (e.g. resnet50) or model.source to a built-in "
            "torchvision name. For ONNX or pickled models, use a .py file path "
            "instead."
        )

    cached = bundled_cache.get(arch)
    if cached is None:
        try:
            weights_enum = models.get_model_weights(arch)
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"data.{_knob_name(side)}: 'model-bundled' could not resolve weights "
                f"for arch {arch!r}: {exc}. Use a path to a .py file instead."
            ) from exc

        default = weights_enum["DEFAULT"]
        preset = default.transforms()
        data_module, model_module = _split_preset(preset)
        description = f"{weights_enum.__name__}.DEFAULT ({default.name})"
        cached = (data_module, model_module, description)
        bundled_cache[arch] = cached

    data_module, model_module, description = cached
    module = data_module if side == "data" else model_module
    return _SideResolved(module=module, origin="model-bundled", description=description)


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
    """Split a torchvision preset into data preprocessing and model input transformation.

    - ``ImageClassification`` → Resize + CenterCrop for the loader, Normalize
      for the model boundary.
    - ``SemanticSegmentation`` → Resize for the loader, Normalize for the
      boundary.
    - ``ObjectDetection`` (and any unrecognised preset that runs as a no-op)
      → both modules ``None``. Detection models normalise internally.
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


@functools.cache
def _preset_wrapper_cls() -> type:
    # Lazy class factory — ``nn.Module`` as a base is evaluated at class-def
    # time, which would force a real ``import torch`` at module load and break
    # the partial-extras-venv contract (see ``raitap.utils.lazy``). The wrapper
    # lifts a torchvision transforms preset to an ``nn.Module`` so it composes
    # with ``nn.Sequential(transform, backbone)``, transparent to autograd /
    # attribution / attacks.

    class _PresetWrapper(nn.Module):
        def __init__(self, preset: Any) -> None:
            super().__init__()
            self.preset = preset

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.preset(x)

    return _PresetWrapper


# ---------------------------------------------------------------------------
# Custom file
# ---------------------------------------------------------------------------


def _resolve_side_custom_file(
    raw_path: str,
    *,
    side: Side,
    acknowledge_exec: bool,
    file_cache: dict[Path, Any],
    hash_cache: dict[Path, str],
) -> _SideResolved:
    knob = _knob_name(side)
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"data.{knob}: file not found at {path}. "
            "Provide a path to a .py file with the matching RAITAP-decorated "
            "factory."
        )
    if path.suffix.lower() != ".py":
        raise ValueError(
            f"data.{knob}: expected a .py file, got {path.suffix!r}. "
            f"Use 'model-bundled' or a Python file path."
        )

    if not _consent_given(acknowledge_exec=acknowledge_exec):
        raise PermissionError(
            f"Refusing to run {raw_path} because it is arbitrary Python code.\n"
            "\n"
            "To allow it, choose one:\n"
            "  - CLI:          re-run with --allow-preprocessing-exec (short: -yp)\n"
            "  - Python API:   pass acknowledge_preprocessing_exec=True to "
            "raitap.run(...)\n"
            "\n"
            "If you only need standard ImageNet preprocessing, use the model's\n"
            "bundled preprocessing instead:\n"
            "    data:\n"
            f"      {knob}: model-bundled"
        )

    file_sha256 = hash_cache.get(path)
    if file_sha256 is None:
        file_sha256 = _sha256_of_file(path)
        hash_cache[path] = file_sha256

    user_module = file_cache.get(path)
    if user_module is None:
        user_module = _import_user_module(path)
        file_cache[path] = user_module
        _reject_legacy_undecorated_factories(user_module, path)

    if side == "data":
        marker_attr = _DATA_PREPROCESSING_FACTORY_ATTR
        factory_kind = "data preprocessing"
        decorator_name = "raitap_preprocessing_factory"
    else:
        marker_attr = _MODEL_INPUT_TRANSFORMATION_FACTORY_ATTR
        factory_kind = "model input transformation"
        decorator_name = "raitap_model_input_transformation_factory"

    module = _build_decorated_user_factory(
        user_module,
        path,
        marker_attr=marker_attr,
        factory_kind=factory_kind,
        decorator_name=decorator_name,
    )
    if module is None:
        raise AttributeError(
            f"{path}: no factory decorated with `@{decorator_name}` found. "
            f"Decorate one zero-argument factory that returns an nn.Module, "
            f"or point data.{knob} elsewhere."
        )

    return _SideResolved(
        module=module,
        origin="custom-file",
        description=f"Custom {factory_kind} file: {raw_path}",
        file_path=path,
        file_sha256=file_sha256,
    )


def _consent_given(*, acknowledge_exec: bool = False) -> bool:
    if acknowledge_exec:
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


def _reject_legacy_undecorated_factories(user_module: Any, path: Path) -> None:
    legacy = [
        name
        for name in _LEGACY_FACTORY_NAMES
        if callable(factory := getattr(user_module, name, None))
        and not getattr(factory, _DATA_PREPROCESSING_FACTORY_ATTR, False)
        and not getattr(factory, _MODEL_INPUT_TRANSFORMATION_FACTORY_ATTR, False)
    ]
    if not legacy:
        return
    names = ", ".join(f"`{name}()`" for name in legacy)
    raise AttributeError(
        f"{path}: fixed-name factories {names} are no longer supported on their own. "
        "Use the RAITAP decorators: decorate an arbitrary zero-argument factory with "
        "`@raitap_preprocessing_factory` for data preprocessing or "
        "`@raitap_model_input_transformation_factory` for model input transformation."
    )


def _build_decorated_user_factory(
    user_module: Any,
    path: Path,
    *,
    marker_attr: str,
    factory_kind: str,
    decorator_name: str,
) -> nn.Module | None:
    factories = [
        (name, factory)
        for name, factory in vars(user_module).items()
        if callable(factory) and getattr(factory, marker_attr, False)
    ]
    if not factories:
        return None
    if len(factories) > 1:
        names = ", ".join(f"`{name}()`" for name, _factory in factories)
        raise ValueError(
            f"{path}: multiple {factory_kind} factories decorated with "
            f"`@{decorator_name}`: {names}. Use exactly one."
        )
    factory_name, factory = factories[0]
    return _call_user_factory(factory, path, factory_name, factory_kind=factory_kind)


def _call_user_factory(
    factory: Any,
    path: Path,
    factory_name: str,
    *,
    factory_kind: str,
) -> nn.Module:
    _validate_factory_signature(factory, factory_name=factory_name, path=path)
    result = factory()
    if not isinstance(result, nn.Module):
        raise TypeError(
            f"{path}: {factory_kind} factory `{factory_name}()` must return "
            f"an nn.Module, got {type(result).__name__}."
        )
    return result


def _validate_factory_signature(factory: Any, *, factory_name: str, path: Path) -> None:
    """Reject preprocessing factories that require positional arguments.

    Builtins / C-extension callables raise ``ValueError`` from
    ``inspect.signature`` — skip them; the subsequent ``factory()`` call
    will surface any genuine arity mismatch with the underlying Python
    ``TypeError``.
    """
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    ]
    if required:
        names = ", ".join(repr(parameter.name) for parameter in required)
        raise TypeError(
            f"{path}: `{factory_name}()` must be callable with no arguments, "
            f"but it requires {names}. Drop the parameters (or give them defaults) "
            f"and re-run. The factory is invoked by the pipeline as "
            f"`{factory_name}()` — there is no place to thread arguments through."
        )


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


def _knob_name(side: Side) -> str:
    return "preprocessing" if side == "data" else "model_input_transformation"


def _reject_shared_module(data_side: _SideResolved, model_side: _SideResolved) -> None:
    if (
        data_side.module is not None
        and model_side.module is not None
        and data_side.module is model_side.module
    ):
        path_hint = data_side.file_path or model_side.file_path
        location = f" in {path_hint}" if path_hint is not None else ""
        raise ValueError(
            f"data preprocessing and model input transformation must return "
            f"separate nn.Module instances{location}. The model input "
            f"transformation may be moved to the model device, while data "
            f"preprocessing runs on CPU image tensors."
        )


def _compose_description(data_side: _SideResolved, model_side: _SideResolved) -> str:
    if data_side.origin == "off" and model_side.origin == "off":
        return "No preprocessing applied"
    parts: list[str] = []
    if data_side.description is not None:
        parts.append(f"data: {data_side.description}")
    if model_side.description is not None:
        parts.append(f"model: {model_side.description}")
    return "; ".join(parts)
