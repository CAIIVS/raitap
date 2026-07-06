"""Walk a composed Hydra config and infer the matching uv extras.

The walker operates on a plain ``dict``/``Mapping`` — callers compose the
Hydra config (or load YAML) and pass it in. Adapter blocks are recognised by
the presence of ``_target_``; bare class names and fully-qualified paths are
both accepted (the class name is taken from the last dotted segment).

Outputs:
    - ``set[str]`` of extras (deduplicated; the CLI sorts at print/render time)
    - mapping from each extra name to a short human-readable origin phrase
      (used by :mod:`raitap.deps.conflicts` to build error messages)

Unknown adapter ``_target_`` values raise :class:`UnknownAdapterTargetError`
rather than emitting a possibly wrong command.
"""

from __future__ import annotations

import difflib
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from raitap.types import ResolvedHardware


class UnknownAdapterTargetError(RuntimeError):
    """Raised when a ``_target_`` class is not in the adapter→extra map."""


# ``ADAPTER_EXTRAS`` is populated by ``AdapterMixin.__init_subclass__`` once an
# adapter module has been imported — but importing adapter modules pulls their
# wrapped third-party libraries (torch / torchmetrics / captum / ...), which is
# precisely what is missing in the partial-extras venv that the deps bootstrap
# is trying to repair. ``scan_adapter_extras`` is the import-free fallback: an
# AST walk over the raitap source tree that harvests every
# ``class Foo(..., extra="bar")`` declaration without executing the module.
# Adding a new adapter therefore remains a single-file change — the AST scan
# picks the new ``extra=`` kwarg up automatically.
from raitap._adapters import ADAPTER_EXTRAS  # noqa: E402
from raitap.deps.static_scan import scan_adapter_extras, scan_backend_extras  # noqa: E402


def _class_name(target: str) -> str:
    return target.rsplit(".", 1)[-1]


def _extra_for_spec(
    extra: str, supported_hardware: frozenset[ResolvedHardware], hardware: ResolvedHardware
) -> str:
    """Resolve a backend's installable extra for the chosen hardware.

    Bare ``extra`` when the runtime ships a single wheel (``supported_hardware``
    empty, e.g. xgboost). Otherwise ``f"{extra}-{suffix}"`` (e.g. ``torch-cpu``),
    or a clear error when the backend has no wheel for ``hardware``.
    """
    if not supported_hardware:
        return extra
    if hardware not in supported_hardware:
        available = sorted(hw.value for hw in supported_hardware)
        raise ValueError(
            f"The {extra!r} backend has no {hardware.value} build "
            f"(available: {available}). Re-run with a supported --hardware."
        )
    return f"{extra}-{hardware.pyproject_extra_suffix}"


def backend_extra(model_source: str, hardware: ResolvedHardware) -> str:
    """Return the uv extra for a model source, resolved from the backend registry.

    The extension -> extra mapping is harvested import-free from the backends'
    ``@register`` decorators (:func:`scan_backend_extras`): accelerator runtimes
    (torch/onnx) split per hardware, single-wheel ones (xgboost) do not.
    Extensionless sources (built-in torchvision names, e.g. ``"resnet50"``, and
    HuggingFace hub ids) and unknown extensions fall back to the torch runtime —
    a truly unsupported file errors later at load time.
    """
    ext = os.path.splitext(model_source)[1].lower()
    spec = scan_backend_extras().get(ext)
    if spec is None:
        return f"torch-{hardware.pyproject_extra_suffix}"
    extra, supported_hardware = spec
    return _extra_for_spec(extra, supported_hardware, hardware)


def backend_extras(model: Mapping[str, Any], hardware: ResolvedHardware) -> dict[str, str]:
    """Return ``{extra: origin}`` for everything a model config needs to run.

    ``backend_extra`` maps ``model.source`` to a single runtime extra by file
    extension. HuggingFace text models are the one case that needs more than
    that: their source is an extensionless hub id (so ``backend_extra`` falls
    back to the torch runtime, which is correct — transformers runs on torch)
    *and* they set ``model.tokenizer``, which additionally requires the
    ``text`` extra (it carries ``transformers``). Both are returned together
    so callers don't have to special-case the tokenizer signal themselves.
    """
    source = model["source"]
    found: dict[str, str] = {}
    _add(found, backend_extra(source, hardware), f"model.source={source} + hardware={hardware}")
    tokenizer = model.get("tokenizer")
    if tokenizer is not None:
        _add(found, "text", f"model.tokenizer={tokenizer!r}")
    return found


def _extra_for_target(target: str) -> str:
    cls = _class_name(target)
    extra = ADAPTER_EXTRAS.get(cls) or scan_adapter_extras().get(cls)
    if extra is not None:
        return extra
    # Two distinct failures share this path: an end-user typo in ``_target_``
    # (common) and an adapter author who forgot ``extra=`` (rare). Speak to
    # both, and list / suggest valid targets instead of only the author hint.
    known = {**scan_adapter_extras(), **ADAPTER_EXTRAS}
    match = difflib.get_close_matches(cls, known, n=1)
    suggestion = f" Did you mean '{match[0]}'?" if match else ""
    known_list = ", ".join(sorted(known)) if known else "(none found)"
    raise UnknownAdapterTargetError(
        f"Unknown adapter _target_ '{target}'.{suggestion}\n"
        f"Known adapters: {known_list}.\n"
        "If this is a typo, fix the _target_. If you are adding a new adapter, "
        'declare its wrapped-library extra via the ``extra="..."`` class kwarg '
        "so deps inference can pick it up."
    )


def _add(extras: dict[str, str], name: str, origin: str) -> None:
    if name not in extras:
        extras[name] = origin


def _walk_section(
    extras: dict[str, str],
    section_key: str,
    section: Any,
) -> None:
    """Apply adapter→extra mapping for top-level transparency/robustness/etc."""
    if section is None:
        return
    if section_key in {"transparency", "robustness"} and isinstance(section, Mapping):
        for adapter_name, adapter_cfg in section.items():
            if not isinstance(adapter_cfg, Mapping):
                continue
            target = adapter_cfg.get("_target_")
            if isinstance(target, str):
                _add(
                    extras,
                    _extra_for_target(target),
                    f"{section_key}.{adapter_name}._target_={target}",
                )
            evaluation = adapter_cfg.get("evaluation")
            if isinstance(evaluation, Mapping):
                eval_target = evaluation.get("_target_")
                if isinstance(eval_target, str):
                    _add(
                        extras,
                        _extra_for_target(eval_target),
                        f"{section_key}.{adapter_name}.evaluation._target_={eval_target}",
                    )
        return
    if isinstance(section, Mapping):
        target = section.get("_target_")
        if isinstance(target, str):
            _add(extras, _extra_for_target(target), f"{section_key}._target_={target}")


def _walk_launcher(extras: dict[str, str], cfg: Mapping[str, Any]) -> None:
    launcher = (
        cfg.get("hydra", {}).get("launcher") if isinstance(cfg.get("hydra"), Mapping) else None
    )
    if not isinstance(launcher, Mapping):
        return
    target = launcher.get("_target_")
    if isinstance(target, str) and "hydra_submitit_launcher" in target:
        _add(extras, "launcher", f"hydra.launcher._target_={target}")


def infer_extras(
    cfg: Mapping[str, Any],
    *,
    hardware: ResolvedHardware,
) -> tuple[set[str], dict[str, str]]:
    """Return ``(extras, origins)`` for a composed Hydra config.

    ``hardware`` must already be resolved by the caller (probe or override).
    Backend extra is selected from ``cfg['model']['source']``.
    """
    extras: dict[str, str] = {}

    model = cfg.get("model") or {}
    if not isinstance(model, Mapping) or not isinstance(model.get("source"), str):
        raise ValueError(
            "raitap-deps needs model.source to pick a torch vs onnx backend "
            "(the file extension drives that choice). Add `model.source: …` "
            "to the config or override it via Hydra (e.g. `model.source=foo.pt`)."
        )
    for name, origin in backend_extras(model, hardware).items():
        _add(extras, name, origin)

    for section_key in ("transparency", "robustness", "reporting", "tracking", "metrics"):
        _walk_section(extras, section_key, cfg.get(section_key))

    _walk_launcher(extras, cfg)

    return set(extras), extras
