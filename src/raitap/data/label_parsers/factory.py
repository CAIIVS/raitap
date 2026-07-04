"""Instantiation factory for label parsers (mirrors metrics/factory.py:44-60)."""

from __future__ import annotations

from typing import Any

from hydra.utils import instantiate

from raitap import raitap_log
from raitap.configs import cfg_to_dict, resolve_target
from raitap.data.label_parsers.base import (
    LabelParser,  # noqa: TC001  must stay runtime-resolvable for get_type_hints()
)

_LABELS_PREFIX = "raitap.data.label_parsers."


def create_label_parser(labels_config: Any) -> LabelParser:
    """Instantiate a label parser from Hydra-style config (``_target_`` + kwargs)."""
    labels_cfg = cfg_to_dict(labels_config)
    target_path: str = labels_cfg.get("_target_", "")
    resolved_target = resolve_target(target_path, _LABELS_PREFIX)
    labels_cfg["_target_"] = resolved_target

    try:
        parser = instantiate(labels_cfg)
    except Exception as e:
        raitap_log.exception("Label parser instantiation failed for target %r", target_path)
        raise ValueError(
            f"Could not instantiate label parser {target_path!r}.\n"
            "Check that _target_ points to a valid LabelParser implementation."
        ) from e

    return parser
