"""Shared use-key-resolve-and-instantiate core for data parser families."""

from __future__ import annotations

from typing import Any

from hydra.utils import instantiate

from raitap import raitap_log
from raitap.configs import cfg_to_dict
from raitap.configs.registry_resolve import reject_config_target, resolve_target_fqn


def create_parser(config: Any, *, group: str, kind: str) -> Any:
    """Instantiate a parser from a ``use:``-keyed config, resolved via the
    trusted registry seam (``raitap.configs.registry_resolve``)."""
    cfg = cfg_to_dict(config)
    reject_config_target(cfg)
    use = cfg.get("use", "")
    fqn = resolve_target_fqn(group, use)
    cfg["_target_"] = fqn
    cfg.pop("use", None)
    try:
        return instantiate(cfg)
    except Exception as e:
        raitap_log.exception("%s instantiation failed for target %r", kind, fqn)
        raise ValueError(
            f"Could not instantiate {kind} {fqn!r}.\nCheck that `use` resolves to a "
            f"valid {kind} implementation."
        ) from e
