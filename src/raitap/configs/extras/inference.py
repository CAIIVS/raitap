"""Walk a composed Hydra config and infer the matching uv extras.

The walker operates on a plain ``dict``/``Mapping`` — callers compose the
Hydra config (or load YAML) and pass it in. Adapter blocks are recognised by
the presence of ``_target_``; bare class names and fully-qualified paths are
both accepted (the class name is taken from the last dotted segment).

Outputs:
    - sorted list of extras (deduplicated)
    - mapping from each extra name to a short human-readable origin phrase
      (used by :mod:`raitap.configs.extras.conflicts` to build error messages)

Unknown adapter ``_target_`` values raise :class:`UnknownAdapterTargetError`
rather than emitting a possibly wrong command.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

Hardware = Literal["cpu", "cuda", "xpu"]


class UnknownAdapterTargetError(RuntimeError):
    """Raised when a ``_target_`` class is not in the adapter→extra map."""


ADAPTER_EXTRAS: dict[str, str] = {
    "CaptumExplainer": "captum",
    "ShapExplainer": "shap",
    "TorchattacksAssessor": "torchattacks",
    "FoolboxAssessor": "foolbox",
    "MarabouAssessor": "marabou",
    "HTMLReporter": "jinja",
    "PDFReporter": "borb",
    "MLFlowTracker": "mlflow",
    "ClassificationMetrics": "metrics",
    "DetectionMetrics": "metrics",
}

_HARDWARE_SUFFIX: dict[Hardware, str] = {"cuda": "cuda", "xpu": "intel", "cpu": "cpu"}


def _class_name(target: str) -> str:
    return target.rsplit(".", 1)[-1]


def backend_extra(model_source: str, hardware: Hardware) -> str:
    """Return ``torch-<hw>`` or ``onnx-<hw>`` based on the source file extension."""
    ext = os.path.splitext(model_source)[1].lower()
    suffix = _HARDWARE_SUFFIX[hardware]
    if ext == ".onnx":
        return f"onnx-{suffix}"
    return f"torch-{suffix}"


def _extra_for_target(target: str) -> str:
    cls = _class_name(target)
    if cls not in ADAPTER_EXTRAS:
        raise UnknownAdapterTargetError(
            f"Unknown adapter _target_ '{target}' — extend "
            "raitap.configs.extras.inference.ADAPTER_EXTRAS if you added a new "
            "library-backed adapter."
        )
    return ADAPTER_EXTRAS[cls]


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
    hardware: Hardware,
) -> tuple[set[str], dict[str, str]]:
    """Return ``(extras, origins)`` for a composed Hydra config.

    ``hardware`` must already be resolved by the caller (probe or override).
    Backend extra is selected from ``cfg['model']['source']``.
    """
    extras: dict[str, str] = {}

    model = cfg.get("model") or {}
    if not isinstance(model, Mapping) or not isinstance(model.get("source"), str):
        raise ValueError(
            "raitap-deps needs model.source to pick a torch/onnx backend. "
            "Provide it in the config or set --hardware explicitly."
        )
    backend = backend_extra(model["source"], hardware)
    _add(extras, backend, f"model.source={model['source']} + hardware={hardware}")

    for section_key in ("transparency", "robustness", "reporting", "tracking", "metrics"):
        _walk_section(extras, section_key, cfg.get(section_key))

    _walk_launcher(extras, cfg)

    return set(extras), extras
