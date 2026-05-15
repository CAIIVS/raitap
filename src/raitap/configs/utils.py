from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .schema import AppConfig


def cfg_to_dict(cfg: Any) -> dict:
    """
    Normalise any config representation to a plain :class:`dict`.

    Handles:
    * ``omegaconf.DictConfig``             (Hydra runtime)
    * Python ``dataclass``                 (structured config)
    * ``types.SimpleNamespace`` / objects  (unit tests)
    * plain ``dict``
    """
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if hasattr(cfg, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    return dict(cfg)


def resolve_target(target: str, prefix: str) -> str:
    if not target:
        return target
    return target if "." in target else f"{prefix}{target}"


def set_output_root(config: Any, output_root: str | Path) -> None:
    if isinstance(config, DictConfig):
        # OmegaConf structured configs reject keys outside the schema; flip
        # struct off just long enough to set the runtime-only marker.
        was_struct = OmegaConf.is_struct(config)
        OmegaConf.set_struct(config, False)
        try:
            config._output_root = str(output_root)
        finally:
            OmegaConf.set_struct(config, was_struct)
        return
    config._output_root = str(output_root)


def resolve_run_dir(
    config: AppConfig | None = None,
    *,
    output_root: str | Path | None = None,
    subdir: str | None = None,
) -> Path:
    """Resolve the current run directory from Hydra or an explicit fallback root."""
    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        if output_root is None:
            resolved_output_root: str | Path = (
                "." if config is None else str(getattr(config, "_output_root", "."))
            )
        else:
            resolved_output_root = output_root
        run_dir = Path(resolved_output_root)
    if subdir:
        return run_dir / subdir
    return run_dir


def register_configs() -> None:
    """Register the ``AppConfig`` dataclass as ``raitap_schema``.

    User and bundled YAMLs reference it via ``defaults: [raitap_schema, _self_]``
    so unset fields inherit dataclass defaults (e.g. ``reporting.sample_selection``)
    and required fields are ``MISSING`` (loud error if omitted).
    """
    cs = ConfigStore.instance()
    cs.store(name="raitap_schema", node=AppConfig)

    # Register hydra-zen group entries (transparency / robustness / metrics /
    # reporting / tracking) that replaced the per-group YAML files.
    from .zen import register_zen_groups

    register_zen_groups()
