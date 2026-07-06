"""Shared _target_-resolve-and-instantiate core for data parser families."""

from __future__ import annotations

from typing import Any

from hydra.utils import instantiate

from raitap import raitap_log
from raitap.configs import cfg_to_dict, resolve_target


def create_parser(config: Any, *, prefix: str, kind: str) -> Any:
    """Instantiate a parser from Hydra-style config (``_target_`` + kwargs)."""
    cfg = cfg_to_dict(config)
    target_path: str = cfg.get("_target_", "")
    cfg["_target_"] = resolve_target(target_path, prefix)
    try:
        return instantiate(cfg)
    except Exception as e:
        raitap_log.exception("%s instantiation failed for target %r", kind, target_path)
        raise ValueError(
            f"Could not instantiate {kind} {target_path!r}.\n"
            f"Check that _target_ points to a valid {kind} implementation."
        ) from e
