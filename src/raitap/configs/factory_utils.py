from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig


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
    return dict[Any, Any](cfg)


def resolve_target(target: str, prefix: str) -> str:
    if not target:
        return target
    return target if "." in target else f"{prefix}{target}"


def resolve_run_dir(config: AppConfig, subdir: str | None = None) -> Path:
    """Resolve the current run directory from Hydra, with config fallback outside Hydra."""
    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        run_dir = Path(config.fallback_output_dir)
    if subdir:
        return run_dir / subdir
    return run_dir
