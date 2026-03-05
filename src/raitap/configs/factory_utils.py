from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def cfg_to_dict(cfg) -> dict:
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
