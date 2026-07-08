from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .schema import AppConfig

# Cached so a single Python process (any number of orchestrator / phase /
# factory calls) writes into ONE timestamped output dir, not a new one per
# call. Pattern matches Hydra's default ``outputs/<YYYY-MM-DD>/<HH-MM-SS>/``.
_PROCESS_FALLBACK_ROOT: Path | None = None


def _process_fallback_root() -> Path:
    global _PROCESS_FALLBACK_ROOT
    if _PROCESS_FALLBACK_ROOT is None:
        now = datetime.now()
        _PROCESS_FALLBACK_ROOT = (
            Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        )
    return _PROCESS_FALLBACK_ROOT


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
    """Resolve the current run directory.

    Priority:

    1. Hydra's runtime ``output_dir`` if Hydra is active (CLI path).
    2. Explicit ``output_root`` kwarg.
    3. ``config._output_root`` if set (programmatic path via
       :func:`set_output_root` / :func:`raitap.run`).
    4. Process-cached ``outputs/<YYYY-MM-DD>/<HH-MM-SS>/`` fallback so direct
       callers (tests, embedded scripts) never dump artefacts at the cwd
       root. The fallback is computed once per process so a multi-phase run
       writes to one shared dir.
    """
    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        if output_root is not None:
            resolved: str | Path = output_root
        else:
            resolved = _process_fallback_root()
            if config is not None:
                with suppress(Exception):
                    # OmegaConf DictConfig raises ConfigAttributeError on
                    # missing keys; plain dataclasses raise AttributeError.
                    # Either way the fallback above stays in effect.
                    candidate = config._output_root  # type: ignore[attr-defined]
                    if candidate is not None:
                        resolved = candidate
        run_dir = Path(resolved)
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
