"""Single trusted seam: map a config ``use`` key to a vetted class FQN.

Config never carries ``_target_``. Selection is a short ``use: <registry_name>``
key; this module resolves it against the closed registry populated at adapter
registration (``raitap._adapters._TARGET_FQN``). Removing ``_target_`` from the
config layer is what closes the arbitrary-callable RCE surface (issue #301).
"""

from __future__ import annotations

from typing import Any


class UnsafeConfigTargetError(ValueError):
    """A config carried a ``_target_`` key, which is no longer accepted."""


def reject_config_target(cfg: dict[str, Any]) -> None:
    if "_target_" in cfg:
        raise UnsafeConfigTargetError(
            "`_target_` is no longer accepted in configs (arbitrary-callable "
            "security surface). Select an implementation with `use: <key>`."
        )


def resolve_target_fqn(group: str, use: str) -> str:
    from raitap._adapters import _TARGET_FQN

    group_map = _TARGET_FQN.get(group, {})
    try:
        return group_map[use]
    except KeyError:
        valid = ", ".join(sorted(group_map))
        raise ValueError(
            f"Unknown {group} key {use!r}. Valid keys: {valid or '(none registered)'}."
        ) from None
