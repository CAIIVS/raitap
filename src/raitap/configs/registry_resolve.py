"""Single trusted seam: map a config ``use`` key to a vetted class FQN.

Config never carries ``_target_``. Selection is a short ``use: <registry_name>``
key; this module resolves it against the closed registry populated at adapter
registration (``raitap._adapters._TARGET_FQN``). Removing ``_target_`` from the
config layer is what closes the arbitrary-callable RCE surface (issue #301).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from hydra.utils import instantiate as _default_instantiate

from raitap import raitap_log


class UnsafeConfigTargetError(ValueError):
    """A config carried a ``_target_`` key, which is no longer accepted."""


def reject_config_target(cfg: Any) -> None:
    """Reject a ``_target_`` key at any depth in ``cfg``.

    ``hydra.utils.instantiate`` recurses, so a nested ``_target_`` (e.g. inside a
    ``constructor`` value) is as dangerous as a top-level one. Scan the whole tree.
    """
    if isinstance(cfg, Mapping):
        if "_target_" in cfg:
            raise UnsafeConfigTargetError(
                "`_target_` is no longer accepted in configs (arbitrary-callable "
                "security surface). Select an implementation with `use: <key>`."
            )
        for value in cfg.values():
            reject_config_target(value)
    elif isinstance(cfg, (list, tuple)):
        for item in cfg:
            reject_config_target(item)


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


def instantiate_partial_from_use(
    cfg: Any,
    *,
    group: str,
    entity: str,
    config: Any,
    instantiate_fn: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    """Resolve ``cfg["use"]`` to a vetted FQN, partially instantiate it, and call
    the result with ``config``.

    Shared by :func:`raitap.reporting.factory.create_report` and
    :func:`raitap.tracking.base_tracker.BaseTracker.create_tracker`, which both
    follow the same "reject ``_target_`` -> resolve ``use`` -> partial-instantiate
    -> call with the app config" shape, differing only in the registry group and
    the noun used in error messages.

    ``cfg`` is the already-``cfg_to_dict``'d sub-config (e.g. ``config.reporting``)
    carrying the ``use`` key. ``group`` is the :data:`raitap._adapters._TARGET_FQN`
    group to resolve ``use`` against (e.g. ``"reporting"``). ``entity`` names the
    instantiated thing for error messages (e.g. ``"reporter"``). ``config`` is the
    full :class:`~raitap.configs.schema.AppConfig` passed to the resolved class's
    constructor. ``instantiate_fn`` defaults to :func:`hydra.utils.instantiate`
    but can be overridden — mirrors :func:`raitap.configs.adapter_factory.instantiate_adapter`,
    letting callers wire in test doubles against their own module-level binding.
    """
    reject_config_target(cfg)
    use = str(cfg.get("use", ""))
    resolved_target = resolve_target_fqn(group, use)

    fn = instantiate_fn if instantiate_fn is not None else _default_instantiate
    try:
        klass = fn({"_target_": resolved_target, "_partial_": True})
        return klass(config)
    except Exception as error:
        raitap_log.exception("%s instantiation failed for target %r", entity.capitalize(), use)
        raise ValueError(
            f"Could not instantiate {entity} {use!r}.\n"
            f"Check that `use` points to a registered {group} adapter."
        ) from error
